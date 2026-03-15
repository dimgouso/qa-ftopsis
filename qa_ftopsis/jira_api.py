from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from qa_ftopsis.config import AppConfig


def _history_source_field(queue_field: str) -> str:
    normalized = queue_field.strip().lower()
    if normalized in {"component_combo", "components_json"}:
        return "component"
    return normalized


def _search_url(config: AppConfig) -> str:
    if not config.jira.api_base_url:
        raise ValueError("jira.api_base_url is required for Jira API fetching")
    return config.jira.api_base_url.rstrip("/") + "/rest/api/2/search"


def _default_jql(config: AppConfig) -> str:
    if config.jira.jql:
        return config.jira.jql
    if not config.jira.project_key:
        raise ValueError("jira.project_key or jira.jql is required for Jira API fetching")
    return (
        f"project={config.jira.project_key} "
        "AND resolutiondate is not EMPTY "
        "AND component is not EMPTY "
        "ORDER BY created DESC"
    )


def _normalize_issue_row(issue: dict) -> dict[str, object]:
    fields = issue.get("fields", {})
    components = fields.get("components") or []
    return {
        "issue_id": str(issue.get("key") or issue.get("id")),
        "instance_id": fields.get("project", {}).get("key", "jira"),
        "project_key": fields.get("project", {}).get("key", "unknown"),
        "summary": fields.get("summary") or "",
        "description": fields.get("description") or "",
        "created_at": fields.get("created"),
        "resolved_at": fields.get("resolutiondate"),
        "priority": (fields.get("priority") or {}).get("name", "medium"),
        "issue_type": (fields.get("issuetype") or {}).get("name", "unknown"),
        "component": components[0].get("name", "") if components else "",
        "components_json": json.dumps([component.get("name", "") for component in components]),
    }


def _history_rows(issue: dict, queue_field: str) -> list[dict[str, object]]:
    issue_id = str(issue.get("key") or issue.get("id"))
    history_field = _history_source_field(queue_field)
    rows: list[dict[str, object]] = []
    for history in issue.get("changelog", {}).get("histories", []):
        author = history.get("author", {}) or {}
        for item in history.get("items", []):
            field_name = str(item.get("field") or "").strip().lower()
            if field_name not in {history_field.lower(), f"{history_field.lower()}s"}:
                continue
            rows.append(
                {
                    "issue_id": issue_id,
                    "change_time": history.get("created"),
                    "field_name": history_field,
                    "old_value": item.get("fromString") or "",
                    "new_value": item.get("toString") or "",
                    "change_author_mask": author.get("name") or author.get("key") or "unknown",
                }
            )
    return rows


def fetch_jira_api_export(config: AppConfig) -> dict[str, str | int]:
    search_url = _search_url(config)
    jql = _default_jql(config)
    output_issues = config.jira.raw_issues_path
    output_history = config.jira.raw_history_path
    if output_issues is None or output_history is None:
        raise ValueError("jira.raw_issues_path and jira.raw_history_path must be configured")

    Path(output_issues).parent.mkdir(parents=True, exist_ok=True)
    Path(output_history).parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    start_at = 0
    page_size = max(1, int(config.jira.page_size))
    max_issues = max(1, int(config.jira.max_issues))
    issue_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []

    while start_at < max_issues:
        batch_size = min(page_size, max_issues - start_at)
        response = session.get(
            search_url,
            params={
                "jql": jql,
                "startAt": start_at,
                "maxResults": batch_size,
                "fields": "summary,description,components,priority,issuetype,created,resolutiondate,project",
                "expand": "changelog",
            },
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
        issues = payload.get("issues", [])
        if not issues:
            break
        for issue in issues:
            issue_rows.append(_normalize_issue_row(issue))
            history_rows.extend(_history_rows(issue, config.jira.queue_field))
        start_at += len(issues)
        if start_at >= payload.get("total", 0):
            break
        time.sleep(0.1)

    pd.DataFrame(issue_rows).to_parquet(output_issues, index=False)
    pd.DataFrame(history_rows).to_parquet(output_history, index=False)
    return {
        "issues_path": str(Path(output_issues).resolve()),
        "history_path": str(Path(output_history).resolve()),
        "num_issues": len(issue_rows),
        "num_history_rows": len(history_rows),
    }
