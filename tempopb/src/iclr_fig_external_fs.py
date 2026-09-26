"""No-follow, descriptor-relative filesystem primitives for ``fig_external``.

The transaction layer supplies state and commit order. This module supplies
the narrow filesystem operations that bind directory identities, create and
inspect regular files relative to retained descriptors, enforce publication
modes, and atomically replace names without following mutable parent paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import os
from pathlib import Path
import secrets
import stat
from typing import BinaryIO


DIRECTORY_MODE = 0o755
PUBLICATION_FILE_MODE = 0o644
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)


@dataclass
class BoundDirectory:
    """One exact directory inode retained through a no-follow descriptor."""

    path: Path
    label: str
    descriptor: int
    device: int
    inode: int

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


def _close_if_open(descriptor: int) -> None:
    if descriptor >= 0:
        os.close(descriptor)


def bind_directory(path: Path, label: str) -> BoundDirectory:
    """Open every path component without following a symlink."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    descriptor = -1
    try:
        descriptor = os.open(absolute.anchor, _DIRECTORY_FLAGS)
        for component in absolute.parts[1:]:
            next_descriptor = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        identity = os.fstat(descriptor)
        if not stat.S_ISDIR(identity.st_mode):
            raise ValueError(f"{label} must be a directory")
        return BoundDirectory(
            absolute, label, descriptor, identity.st_dev, identity.st_ino
        )
    except (FileNotFoundError, NotADirectoryError) as error:
        _close_if_open(descriptor)
        raise ValueError(
            f"{label} and every parent must be an existing non-symlink directory"
        ) from error
    except OSError as error:
        _close_if_open(descriptor)
        if error.errno in {errno.ELOOP, errno.EMLINK, errno.ENOTDIR}:
            raise ValueError(
                f"{label} must not contain a symlinked or non-directory parent"
            ) from error
        raise
    except BaseException:
        _close_if_open(descriptor)
        raise


def bind_child_directory(
    parent: BoundDirectory, name: str, path: Path, label: str
) -> BoundDirectory:
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent.descriptor)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.EMLINK, errno.ENOTDIR}:
            raise ValueError(f"{label} must not be a symlink") from error
        raise
    identity = os.fstat(descriptor)
    if not stat.S_ISDIR(identity.st_mode):
        os.close(descriptor)
        raise ValueError(f"{label} must be a directory")
    return BoundDirectory(path, label, descriptor, identity.st_dev, identity.st_ino)


def revalidate_directory(bound: BoundDirectory) -> None:
    """Require the retained inode and the no-follow pathname to still agree."""
    if bound.descriptor < 0:
        raise ValueError(f"{bound.label} descriptor is closed")
    retained = os.fstat(bound.descriptor)
    if (
        not stat.S_ISDIR(retained.st_mode)
        or retained.st_dev != bound.device
        or retained.st_ino != bound.inode
    ):
        raise ValueError(f"{bound.label} descriptor identity changed")
    try:
        current = bind_directory(bound.path, bound.label)
    except ValueError as error:
        raise ValueError(
            f"{bound.label} changed after capture or preflight, or contains a symlink"
        ) from error
    try:
        if (current.device, current.inode) != (bound.device, bound.inode):
            raise ValueError(f"{bound.label} changed after preflight")
    finally:
        current.close()


def revalidate_bindings(*bindings: BoundDirectory | None) -> None:
    for binding in bindings:
        if binding is not None:
            revalidate_directory(binding)


def stat_at(parent: BoundDirectory, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def list_names(parent: BoundDirectory) -> tuple[str, ...]:
    return tuple(os.listdir(parent.descriptor))


def validate_regular_or_absent(
    parent: BoundDirectory, name: str, label: str
) -> None:
    identity = stat_at(parent, name)
    if identity is None:
        return
    if stat.S_ISLNK(identity.st_mode):
        raise ValueError(f"{label} must not be a symlink: {name}")
    if not stat.S_ISREG(identity.st_mode):
        raise ValueError(f"{label} must be a regular file: {name}")


def require_regular(parent: BoundDirectory, name: str, label: str) -> None:
    identity = stat_at(parent, name)
    if identity is None:
        raise ValueError(f"partial analysis state: missing {name}")
    if stat.S_ISLNK(identity.st_mode):
        raise ValueError(f"{label} must not be a symlink: {name}")
    if not stat.S_ISREG(identity.st_mode):
        raise ValueError(f"{label} must be a regular file: {name}")


def _capture_identity(identity: os.stat_result) -> tuple[int, ...]:
    return (
        identity.st_dev,
        identity.st_ino,
        identity.st_size,
        identity.st_mtime_ns,
        identity.st_ctime_ns,
    )


def read_regular(parent: BoundDirectory, name: str) -> bytes:
    before = stat_at(parent, name)
    if before is None or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"required regular file is missing or nonregular: {name}")
    descriptor = os.open(name, os.O_RDONLY | _FILE_NOFOLLOW, dir_fd=parent.descriptor)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"file changed before descriptor capture: {name}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = stat_at(parent, name)
    if current is None or _capture_identity(after) != _capture_identity(current):
        raise ValueError(f"file changed during descriptor capture: {name}")
    return b"".join(chunks)


def _create_temp_descriptor(parent: BoundDirectory, name: str) -> tuple[int, str]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FILE_NOFOLLOW
    for _ in range(100):
        temporary = f".{name}.{secrets.token_hex(8)}.tmp"
        try:
            descriptor = os.open(
                temporary, flags, PUBLICATION_FILE_MODE, dir_fd=parent.descriptor
            )
        except FileExistsError:
            continue
        os.fchmod(descriptor, PUBLICATION_FILE_MODE)
        return descriptor, temporary
    raise FileExistsError(f"could not allocate a unique staged file for {name}")


def write_regular_temp(
    parent: BoundDirectory,
    name: str,
    content: bytes,
    *,
    revalidate: bool = True,
    mode: int = PUBLICATION_FILE_MODE,
) -> str:
    if revalidate:
        revalidate_directory(parent)
    descriptor, temporary = _create_temp_descriptor(parent, name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        unlink_regular_if_present(parent, temporary)
        raise
    require_regular(parent, temporary, "staged output")
    return temporary


def open_staged_output(parent: BoundDirectory, name: str) -> tuple[BinaryIO, str]:
    revalidate_directory(parent)
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FILE_NOFOLLOW,
        PUBLICATION_FILE_MODE,
        dir_fd=parent.descriptor,
    )
    os.fchmod(descriptor, PUBLICATION_FILE_MODE)
    return os.fdopen(descriptor, "wb"), name


def open_regular_temp(parent: BoundDirectory, name: str) -> tuple[BinaryIO, str]:
    revalidate_directory(parent)
    descriptor, temporary = _create_temp_descriptor(parent, name)
    return os.fdopen(descriptor, "wb"), temporary


def finish_staged_output(
    parent: BoundDirectory, name: str, handle: BinaryIO
) -> None:
    handle.flush()
    os.fsync(handle.fileno())
    os.fchmod(handle.fileno(), PUBLICATION_FILE_MODE)
    handle.close()
    require_regular(parent, name, "staged output")


def unlink_regular_if_present(parent: BoundDirectory, name: str | None) -> None:
    if name is None:
        return
    identity = stat_at(parent, name)
    if identity is None:
        return
    if not stat.S_ISREG(identity.st_mode) or stat.S_ISLNK(identity.st_mode):
        raise RuntimeError(f"transaction cleanup found a nonregular entry: {name}")
    os.unlink(name, dir_fd=parent.descriptor)


def replace_retained_at(
    source_parent: BoundDirectory,
    source_name: str,
    destination_parent: BoundDirectory,
    destination_name: str,
) -> None:
    os.replace(
        source_name,
        destination_name,
        src_dir_fd=source_parent.descriptor,
        dst_dir_fd=destination_parent.descriptor,
    )


def replace_at(
    source_parent: BoundDirectory,
    source_name: str,
    destination_parent: BoundDirectory,
    destination_name: str,
    *,
    bindings: tuple[BoundDirectory, ...],
) -> None:
    revalidate_bindings(*bindings, source_parent, destination_parent)
    replace_retained_at(
        source_parent, source_name, destination_parent, destination_name
    )
    revalidate_bindings(*bindings, destination_parent)


def create_bound_directory(
    parent: BoundDirectory, name: str, path: Path, label: str
) -> BoundDirectory:
    os.mkdir(name, DIRECTORY_MODE, dir_fd=parent.descriptor)
    bound = bind_child_directory(parent, name, path, label)
    os.fchmod(bound.descriptor, DIRECTORY_MODE)
    return bound


def set_directory_mode(bound: BoundDirectory, mode: int = DIRECTORY_MODE) -> None:
    os.fchmod(bound.descriptor, mode)


def remove_empty_directory(parent: BoundDirectory, name: str) -> None:
    os.rmdir(name, dir_fd=parent.descriptor)
