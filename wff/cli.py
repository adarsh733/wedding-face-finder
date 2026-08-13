"""Command line for Stage 1 and Stage 2.

    python -m wff manifest  <event-id> <link>    walk the source, write the list
    python -m wff process   <event-id> <link>    the expensive stage, resumable
    python -m wff cluster   <event-id>           the cheap stage + HTML report
    python -m wff run       <event-id> <link>    all three
    python -m wff ui                             the testing console in a browser
    python -m wff verify-drive <link>            open item #1, on its own
    python -m wff set-key   <google-api-key>     once, ever -- unlocks Drive links

No database, no cloud, no login. Week 1-2 of the build order.
"""
from __future__ import annotations

import argparse
import sys

from .config import load_config
from .ingest import adapter_for
from .storage import get_store


def _print_stats(stats, entries) -> None:
    print(f"\nFound:   {stats.images:,} images  ·  {stats.total_bytes / 1e9:.1f} GB")
    if stats.sub_albums:
        albums = " · ".join(
            f"{name} {count:,}"
            for name, count in sorted(stats.sub_albums.items(), key=lambda kv: -kv[1])
        )
        print(f"         {albums}")
    skipped = []
    if stats.videos:
        skipped.append(f"{stats.videos} videos")
    if stats.raw:
        skipped.append(f"{stats.raw} RAW files")
    if stats.other:
        skipped.append(f"{stats.other} other files")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")
    # ~3.5s/photo on a desktop CPU with tiling on. Re-measure once we have run
    # a real wedding; this is an estimate, not a promise.
    print(f"Estimated time: about {len(entries) * 3.5 / 3600:.1f} hours\n")


def cmd_manifest(args) -> int:
    cfg = load_config()
    store = get_store(cfg)
    adapter = adapter_for(args.link)

    check = adapter.validate()
    print(f"[{adapter.source_type}] {check.message}")
    if not check.ok:
        return 1

    from .stage1 import build_manifest

    entries, stats = build_manifest(adapter, args.event_id, store, cfg)
    _print_stats(stats, entries)
    if not entries:
        print("No images found. Nothing to do.")
        return 1
    return 0


def cmd_process(args) -> int:
    import time
    from datetime import datetime, timezone

    cfg = load_config()
    store = get_store(cfg)
    adapter = adapter_for(args.link)

    from .runs import RunRecord, append_run, human_duration
    from .stage1 import EventPaths, build_manifest, consolidate, process_event

    # Timed and journalled exactly like a console run. If only the console
    # recorded runs, work done from the terminal would be missing from every
    # dashboard total.
    wall_start = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    reviewer = getattr(args, "reviewer", None) or "cli"

    paths = EventPaths(cfg.storage.bucket, args.event_id)
    if not store.exists(paths.manifest):
        check = adapter.validate()
        print(f"[{adapter.source_type}] {check.message}")
        if not check.ok:
            return 1
        entries, stats = build_manifest(adapter, args.event_id, store, cfg)
        _print_stats(stats, entries)

    summary = process_event(
        adapter,
        args.event_id,
        store=store,
        config=cfg,
        limit=args.limit,
        resume=not args.no_resume,
    )
    face_count, photo_count = consolidate(args.event_id, store, cfg)

    total_seconds = time.time() - wall_start
    append_run(
        store,
        cfg.storage.bucket,
        RunRecord(
            event_id=args.event_id,
            reviewer=reviewer,
            trigger="cli",
            phase="stopped" if summary.stopped_early else "done",
            started_at=started_at,
            total_seconds=max(0.01, round(total_seconds, 2)),
            processing_seconds=round(summary.elapsed_seconds, 2),
            photos_total=summary.photos_total,
            photos_processed=summary.photos_processed,
            photos_skipped_done=summary.photos_skipped_done,
            photos_failed=summary.photos_failed,
            faces_detected=summary.faces_detected,
            faces_accepted=summary.faces_accepted,
            source_type=adapter.source_type,
            model_version=cfg.embedding_model_version,
        ),
    )
    print(f"\ntotal time         {human_duration(total_seconds)}")

    print(f"\n{'-' * 58}")
    print(f"photos processed   {summary.photos_processed:,}")
    if summary.photos_skipped_done:
        print(f"already done       {summary.photos_skipped_done:,} (resumed)")
    if summary.photos_failed:
        print(f"FAILED             {summary.photos_failed:,}")
    print(f"faces detected     {summary.faces_detected:,}")
    print(
        f"faces accepted     {summary.faces_accepted:,} "
        f"({summary.faces_second_class:,} second-class 50-80px)"
    )
    for reason, count in sorted(summary.reject_reasons.items(), key=lambda kv: -kv[1]):
        print(f"  rejected: {reason:<22} {count:,}")
    if summary.elapsed_seconds and summary.photos_processed:
        per_photo = summary.elapsed_seconds / summary.photos_processed
        print(f"speed              {per_photo:.2f} s/photo")
    print(f"cached             {face_count:,} face rows, {photo_count:,} photo rows")
    return 0


def cmd_cluster(args) -> int:
    cfg = load_config()
    store = get_store(cfg)

    from .report import build_report
    from .stage2 import run_stage2

    output = run_stage2(args.event_id, store, cfg)
    result = output.result

    print(f"\n{'-' * 58}")
    print(f"accepted faces     {len(output.faces):,}")
    print(f"pass 1 (tight)     {result.pass1_group_count} groups")
    print(f"pass 2 (merge)     {result.pass2_merge_count} merges applied")
    if result.blocked_merges_same_photo:
        print(
            f"  same-photo rule blocked {result.blocked_merges_same_photo} merges "
            "(prevented catastrophes)"
        )
    print(f"PEOPLE FOUND       {len(result.persons)}")
    print(f"leftover faces     {result.leftover_face_count:,} (kept, person_id NULL)")
    if result.second_class_assigned:
        print(f"second-class joined {result.second_class_assigned:,}")

    if output.evaluation:
        print(f"\n{'-' * 58}")
        print("SCORED AGAINST GROUND TRUTH")
        for line in output.evaluation.summary_lines():
            print(f"  {line}")

    for alarm in output.sanity.alarms:
        print(f"\n!! {alarm}")
    for note in output.sanity.notes:
        print(f"\n · {note}")

    report_uri = build_report(output, store, cfg)
    print(f"\nreport: {store.public_url(report_uri)}")
    return 0


def cmd_run(args) -> int:
    code = cmd_process(args)
    if code != 0:
        return code
    return cmd_cluster(args)


def cmd_ui(args) -> int:
    """The testing console: paste a folder, watch it run, review every pile.

    Binds to 127.0.0.1 by default -- this machine only. Face data is biometric
    data under the DPDP Act (landmine #3), so it does not go on a network
    because a flag was easy to add. Use --host to override deliberately.
    """
    try:
        from .web import create_app
    except ImportError as exc:
        print(f"The console needs Flask: pip install flask   ({exc})")
        return 1

    cfg = load_config()
    app = create_app(cfg)

    # 127.0.0.1, not the prettier "localhost". On Windows, localhost resolves to
    # ::1 first and the server binds IPv4 only, so every connection waits ~2s for
    # the IPv6 attempt to be refused before falling back. Browsers hide it by
    # racing both; anything scripted -- curl, a health check, a timing probe --
    # pays it in full and reports a 2-second page that is really 30 ms.
    url = f"http://{args.host}:{args.port}/"
    print(f"\n  Testing console: {url}")
    print("  Paste a folder path on that page and press Start.")
    if args.host != "127.0.0.1":
        print(
            "\n  WARNING: bound to "
            f"{args.host}, so anyone who can reach this machine can read the "
            "faces. Face data is biometric data -- do not leave this running."
        )
    print("\n  Ctrl+C to stop.\n")

    if not args.no_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    # threaded: a 4-hour Stage 1 job runs in a background thread and the page
    # must still answer progress polls while it works.
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


def cmd_verify_drive(args) -> int:
    """Open item #1: prove the anonymous-public-link path works end to end.

    Until this passes against a real folder, the "no OAuth, no restricted
    scope, no audit" finding is a hypothesis, not a fact.
    """
    from .ingest.gdrive import GoogleDriveFolderAdapter

    adapter = GoogleDriveFolderAdapter(args.link)
    check = adapter.validate()
    print(f"validate: {'OK' if check.ok else 'FAILED'} -- {check.message}")
    if not check.ok:
        return 1

    print("walking the folder (anonymous, API key only, no OAuth)...")
    refs, stats = adapter.scan()
    print(f"  {stats.images} images, {stats.videos} videos, {stats.raw} RAW")
    if stats.sub_albums:
        print(f"  sub-albums: {dict(stats.sub_albums)}")
    if not refs:
        print("FAILED: no images found in that folder.")
        return 1

    first = refs[0]
    print(f"streaming one file to check download works: {first.path}")
    with adapter.open(first) as stream:
        head = stream.read(65536)
    print(f"  read {len(head):,} bytes, starts with {head[:4]!r}")
    print(
        "\nVERIFIED: a public Drive folder is readable anonymously with an API "
        "key.\nLandmine #2 is defused for real, not just on paper."
    )
    return 0


def cmd_set_key(args) -> int:
    """Save the Google key once, on this machine, outside the repo.

    It used to be an environment variable only, so it vanished with the
    terminal window and the console -- started from a shortcut -- never saw it
    at all. That is why a Drive link failed in under a second.
    """
    from .config import save_google_api_key
    from .ingest.gdrive import GoogleDriveFolderAdapter

    key = args.key.strip()
    if not key:
        print("Nothing to save -- pass the key: python -m wff set-key AIza...")
        return 1
    if key.startswith("http") or "/" in key:
        print("That looks like a link, not a key. The key is a long string of")
        print("letters and numbers, usually starting with AIza.")
        return 1

    path = save_google_api_key(key)
    # Show only the ends. The key is not a secret worth much, but a full key
    # printed to a terminal ends up in screenshots and scrollback.
    print(f"Saved {key[:6]}...{key[-4:]} to {path}")

    if getattr(args, "check", None):
        adapter = GoogleDriveFolderAdapter(args.check)
        check = adapter.validate()
        print(f"\n{'OK' if check.ok else 'NOT WORKING YET'} -- {check.message}")
        return 0 if check.ok else 1

    print("\nNow try it:  python -m wff verify-drive <your-folder-link>")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wff", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_event(p, with_link: bool = True):
        p.add_argument("event_id", help="e.g. ev_sharma01")
        if with_link:
            p.add_argument("link", help="Drive folder link, or a local folder path")

    p_manifest = sub.add_parser("manifest", help="walk the source, write the list")
    add_event(p_manifest)
    p_manifest.set_defaults(func=cmd_manifest)

    p_process = sub.add_parser("process", help="stage 1: detect + embed (resumable)")
    add_event(p_process)
    p_process.add_argument("--limit", type=int, default=None, help="stop after N photos")
    p_process.add_argument("--no-resume", action="store_true", help="reprocess everything")
    p_process.add_argument("--reviewer", default=None, help="who ran this (adarsh/devesh)")
    p_process.set_defaults(func=cmd_process)

    p_cluster = sub.add_parser("cluster", help="stage 2: cluster + HTML report")
    add_event(p_cluster, with_link=False)
    p_cluster.set_defaults(func=cmd_cluster)

    p_run = sub.add_parser("run", help="manifest + process + cluster")
    add_event(p_run)
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--no-resume", action="store_true")
    p_run.add_argument("--reviewer", default=None, help="who ran this (adarsh/devesh)")
    p_run.set_defaults(func=cmd_run)

    p_ui = sub.add_parser("ui", help="the testing console, in a browser")
    p_ui.add_argument("--port", type=int, default=8765)
    p_ui.add_argument(
        "--host",
        default="127.0.0.1",
        help="0.0.0.0 exposes it to the network -- see the DPDP warning",
    )
    p_ui.add_argument("--no-browser", action="store_true", help="don't open a tab")
    p_ui.set_defaults(func=cmd_ui)

    p_drive = sub.add_parser("verify-drive", help="open item #1: prove the Drive path")
    p_drive.add_argument("link")
    p_drive.set_defaults(func=cmd_verify_drive)

    p_key = sub.add_parser("set-key", help="save the Google key on this machine, once")
    p_key.add_argument("key", help="the API key from console.cloud.google.com")
    p_key.add_argument(
        "--check", metavar="LINK", default=None, help="test it against a folder link"
    )
    p_key.set_defaults(func=cmd_set_key)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
