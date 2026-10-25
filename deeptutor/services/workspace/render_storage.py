"""Inspect and clean reproducible Manim files without touching published media."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil

from .models import WorkspaceError
from .service import ContentWorkspaceService


def _render_paths(
    service: ContentWorkspaceService, workspace_id: str, item_id: str
) -> tuple[Path, Path]:
    _, item = service.resolve_published_item(workspace_id, item_id)
    parts = PurePosixPath(item.relative_path).parts
    if (
        not item.generated
        or len(parts) != 6
        or parts[:2] != ("outputs", "math_animator")
        or parts[4] != "artifacts"
        or any(part in {".", ".."} for part in parts)
    ):
        raise WorkspaceError("Only published Math Animator artifacts support render cleanup.")
    binding = service.binding_by_id(workspace_id)
    cursor = binding.root
    for part in parts:
        cursor /= part
        if cursor.is_symlink():
            raise WorkspaceError("Render storage cannot contain symbolic links.")
    artifact = service.resolve(binding, item.relative_path, write=True)
    if not artifact.is_file():
        raise WorkspaceError("The original artifact is unavailable; render files were preserved.")
    return artifact, artifact.parent.parent / "media"


def _inventory(media: Path) -> dict:
    if media.is_symlink():
        raise WorkspaceError("Render storage cannot contain symbolic links.")
    digest = hashlib.sha256()
    size = count = 0
    if media.exists():
        if not media.is_dir():
            raise WorkspaceError("Render storage is not a directory.")
        for parent, directories, files in os.walk(media, followlinks=False):
            for name in sorted(directories + files):
                path = Path(parent) / name
                if path.is_symlink():
                    raise WorkspaceError("Render storage cannot contain symbolic links.")
                stat = path.stat()
                digest.update(
                    f"{path.relative_to(media)}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
                )
                if path.is_file():
                    size += stat.st_size
                    count += 1
            directories.sort()
    return {"cache_bytes": size, "cache_files": count, "fingerprint": digest.hexdigest()}


def inspect_render_storage(
    service: ContentWorkspaceService, workspace_id: str, item_id: str
) -> dict:
    """Describe one published artifact and its rebuildable render directory."""
    artifact, media = _render_paths(service, workspace_id, item_id)
    return {
        "file_path": str(artifact),
        "file_bytes": artifact.stat().st_size,
        "cache_path": str(media),
        **_inventory(media),
    }


def clean_render_storage(
    service: ContentWorkspaceService, workspace_id: str, item_id: str, fingerprint: str
) -> dict:
    """Delete only the unchanged media directory from an approved preview."""
    artifact, media = _render_paths(service, workspace_id, item_id)
    preview = _inventory(media)
    if preview["fingerprint"] != fingerprint:
        raise WorkspaceError("Render files changed. Review the storage preview again.")
    # media 包含分段、字体缓存和渲染副本；最终产物位于相邻的 artifacts 目录。
    if media.exists():
        shutil.rmtree(media)
    return {
        "file_path": str(artifact),
        "freed_bytes": preview["cache_bytes"],
        "removed_files": preview["cache_files"],
        **_inventory(media),
    }
