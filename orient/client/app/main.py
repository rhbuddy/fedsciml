"""Orient Client CLI.

Example::

    python -m app.main --server http://127.0.0.1:8000 --client-id client-1 \
        --dataset examples/bundles/poisson_client1

The client registers, then loops: poll assignment -> download global weights ->
train locally -> upload weights (only). Raw data never leaves the machine.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional, Sequence

from .dataset import load_bundle
from .protocol import ClientRegistration
from .trainer import LocalTrainer
from .transport import ServerClient, ServerConnectionError
from .weights import bytes_to_state_dict, state_dict_to_bytes

logger = logging.getLogger("orient.client")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="orient-client",
        description="Orient federated SciML client: train locally, upload weights only.",
    )
    parser.add_argument("--server", required=True, help="Server base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--client-id", required=True, help="Unique id for this client")
    parser.add_argument("--dataset", required=True, help="Path to the dataset bundle directory")
    parser.add_argument("--problem", default=None, help="Problem name (defaults to dataset.json)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between polls")
    parser.add_argument("--grad-clip", type=float, default=None, help="Max gradient norm (optional)")
    parser.add_argument("--once", action="store_true", help="Train one round, then exit")
    parser.add_argument("--max-wait", type=float, default=0.0, help="Stop waiting after N seconds (0 = forever)")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    bundle = load_bundle(args.dataset, problem=args.problem)
    logger.info(
        "Local dataset ready: problem=%s samples=%d (%s)",
        bundle.problem, bundle.size, bundle.path,
    )

    trainer = LocalTrainer(bundle.problem)
    rounds_done = 0
    waited_since: Optional[float] = None
    started_at = time.time()

    try:
        with ServerClient(args.server) as server:
            server.health()
            info = server.register(
                ClientRegistration(
                    client_id=args.client_id,
                    problem=bundle.problem,
                    n_samples=bundle.size,
                    domain=bundle.domain,
                )
            )
            logger.info("Registered '%s' with %s", info.client_id, args.server)

            while True:
                resp = server.assignment(args.client_id)

                if resp.status == "error":
                    logger.error("Server error: %s", resp.message)
                    return 1
                if resp.status == "done":
                    if rounds_done == 0:
                        # We connected after a previous run had already finished.
                        # Exiting now would leave a stale entry in the server's
                        # expected set and stall the NEXT run, so wait instead.
                        logger.warning(
                            "Server reports the previous run as complete and we have "
                            "trained 0 rounds - waiting for a new run to start."
                        )
                        if args.max_wait > 0 and time.time() - started_at > args.max_wait:
                            logger.warning("--max-wait elapsed with no new run; exiting.")
                            return 2
                        time.sleep(args.poll_interval)
                        continue
                    logger.info("Run complete. Rounds trained: %d", rounds_done)
                    return 0
                if resp.status == "wait":
                    if args.once and rounds_done > 0:
                        logger.info("--once: exiting after %d round(s).", rounds_done)
                        return 0
                    now = time.time()
                    waited_since = waited_since or now
                    if args.max_wait > 0 and now - waited_since > args.max_wait:
                        logger.warning("--max-wait elapsed with no assignment; exiting.")
                        return 2
                    time.sleep(args.poll_interval)
                    continue

                # status == "train"
                if resp.model_spec is None or resp.problem_spec is None or resp.optimizer is None:
                    logger.error("Malformed assignment from server: %s", resp.model_dump())
                    return 1

                waited_since = None
                trainer.assert_spec_compatible(resp.model_spec)
                raw = server.download_weights(args.client_id, resp.round)
                trainer.load_weights(bytes_to_state_dict(raw))

                logger.info(
                    "Round %d/%d: training %d epoch(s) on %d samples (clip=%s)",
                    resp.round, resp.total_rounds, resp.local_epochs, bundle.size, resp.gradient_clip,
                )
                # SRS gradient clipping: server config takes precedence over CLI legacy flag
                stats = trainer.train(
                    bundle.arrays,
                    epochs=resp.local_epochs,
                    optimizer_spec=resp.optimizer,
                    batch_size=args.batch_size,
                    prox_mu=resp.prox_mu,
                    grad_clip=args.grad_clip,
                    gradient_clip=resp.gradient_clip,
                    max_norm=resp.max_norm,
                    clip_value=resp.clip_value,
                )
                logger.info(
                    "Round %d: local loss %.6g (%d steps)",
                    resp.round, stats["loss"], stats["steps"],
                )

                server.upload_weights(
                    args.client_id,
                    resp.round,
                    bundle.size,
                    stats["loss"],
                    state_dict_to_bytes(trainer.state_dict()),
                )
                rounds_done += 1

                if args.once:
                    logger.info("--once: exiting after 1 round.")
                    return 0

    except ServerConnectionError as exc:
        logger.error("%s", exc)
        return 3
    except ValueError as exc:
        logger.error("Incompatible assignment: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Rounds trained: %d", rounds_done)
        return 0


if __name__ == "__main__":
    sys.exit(main())
