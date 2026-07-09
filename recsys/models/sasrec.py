from __future__ import annotations


class SASRecUnavailableError(RuntimeError):
    pass


def build_sasrec_placeholder(*_args, **_kwargs):
    raise SASRecUnavailableError(
        "SASRec is intentionally staged after baselines. Install Torch and implement the sequence "
        "encoder once session metrics justify the additional serving complexity."
    )
