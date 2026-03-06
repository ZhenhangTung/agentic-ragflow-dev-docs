"""
Download RAGFlow reference documentation from GitHub.
"""
import os
from datetime import datetime, timedelta, timezone
import httpx
from rich.console import Console
from src.config import get_settings

console = Console()


async def download_docs(force: bool = False) -> list[str]:
    """
    Download reference docs from the RAGFlow GitHub repo.
    Returns list of local file paths.
    """
    settings = get_settings()
    docs_dir = settings.docs_dir
    os.makedirs(docs_dir, exist_ok=True)

    downloaded = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for filename in settings.doc_files:
            local_path = os.path.join(docs_dir, filename)

            if os.path.exists(local_path) and not force:
                console.print(f"  [dim]Skipping {filename} (already exists)[/dim]")
                downloaded.append(local_path)
                continue

            url = f"{settings.github_raw_base}/{filename}"
            console.print(f"  [cyan]Downloading {filename}...[/cyan]")

            try:
                resp = await client.get(url)
                resp.raise_for_status()
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                downloaded.append(local_path)
                console.print(f"  [green]✓ {filename} ({len(resp.text)} chars)[/green]")
            except httpx.HTTPError as e:
                console.print(f"  [red]✗ Failed to download {filename}: {e}[/red]")

        issues_doc = await sync_github_issues(client=client, force=force)
        if issues_doc:
            downloaded.append(issues_doc)

    return downloaded


def _should_refresh(local_path: str, sync_hours: int, force: bool) -> bool:
    if force or not os.path.exists(local_path):
        return True
    if sync_hours <= 0:
        return True

    modified_at = datetime.fromtimestamp(os.path.getmtime(local_path), tz=timezone.utc)
    return datetime.now(timezone.utc) - modified_at >= timedelta(hours=sync_hours)


def _format_issues_markdown(owner: str, repo: str, issues: list[dict]) -> str:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"# GitHub Issues Snapshot ({owner}/{repo})",
        "",
        f"Generated at: {generated_at}",
        f"Total issues: {len(issues)}",
        "",
    ]

    if not issues:
        lines.extend(["No issues found.", ""])
        return "\n".join(lines)

    for issue in issues:
        issue_number = issue.get("number", "")
        title = (issue.get("title") or "").strip()
        state = issue.get("state", "unknown")
        updated_at = issue.get("updated_at", "")
        created_at = issue.get("created_at", "")
        closed_at = issue.get("closed_at") or "-"
        labels = ", ".join(label.get("name", "") for label in issue.get("labels", [])) or "-"
        body = (issue.get("body") or "").strip() or "_No description provided._"

        lines.extend(
            [
                f"## Issue #{issue_number}: {title}",
                "",
                "### Metadata",
                f"- State: {state}",
                f"- Labels: {labels}",
                f"- Created at: {created_at}",
                f"- Updated at: {updated_at}",
                f"- Closed at: {closed_at}",
                f"- URL: {issue.get('html_url', '')}",
                "",
                "### Body",
                body,
                "",
            ]
        )
    return "\n".join(lines)


async def sync_github_issues(
    client: httpx.AsyncClient,
    force: bool = False,
) -> str | None:
    settings = get_settings()
    if not settings.github_issues_enabled:
        return None

    os.makedirs(settings.docs_dir, exist_ok=True)
    local_path = os.path.join(settings.docs_dir, settings.github_issues_filename)

    if not _should_refresh(local_path, settings.github_issues_sync_hours, force):
        console.print(
            f"  [dim]Skipping {settings.github_issues_filename} "
            f"(updated within {settings.github_issues_sync_hours}h)[/dim]"
        )
        return local_path

    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"

    api_url = f"https://api.github.com/repos/{settings.github_issues_owner}/{settings.github_issues_repo}/issues"
    per_page = max(1, min(settings.github_issues_per_page, 100))

    issues: list[dict] = []
    console.print(
        f"  [cyan]Syncing GitHub issues from {settings.github_issues_owner}/{settings.github_issues_repo}...[/cyan]"
    )
    try:
        for page in range(1, max(1, settings.github_issues_max_pages) + 1):
            resp = await client.get(
                api_url,
                params={
                    "state": settings.github_issues_state,
                    "sort": "updated",
                    "direction": "desc",
                    "per_page": per_page,
                    "page": page,
                },
                headers=headers,
            )
            resp.raise_for_status()
            page_items = resp.json()
            if not page_items:
                break

            issues.extend(item for item in page_items if "pull_request" not in item)
            if len(page_items) < per_page:
                break
    except httpx.HTTPError as e:
        console.print(f"  [red]✗ Failed to sync GitHub issues: {e}[/red]")
        if os.path.exists(local_path):
            return local_path
        return None

    content = _format_issues_markdown(
        settings.github_issues_owner,
        settings.github_issues_repo,
        issues,
    )
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(content)
    console.print(
        f"  [green]✓ {settings.github_issues_filename} synced ({len(issues)} issues)[/green]"
    )
    return local_path


def list_local_docs() -> list[str]:
    """List locally available doc files."""
    settings = get_settings()
    docs_dir = settings.docs_dir
    if not os.path.exists(docs_dir):
        return []
    return [
        os.path.join(docs_dir, f)
        for f in os.listdir(docs_dir)
        if f.endswith((".md", ".mdx"))
    ]
