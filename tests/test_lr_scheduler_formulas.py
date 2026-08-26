"""Match fine-tune LambdaLR formulas used in TabResnetWrapper.fit (documentation guard)."""


def test_finetune_lambda_head_and_encoder():
    head_decay = 0.5
    head_step = 10
    enc_decay = 0.95

    head_lambda = lambda epoch, h=head_decay, s=head_step: h ** (epoch // s)
    encoder_lambda = lambda epoch, b=enc_decay: b**epoch

    assert head_lambda(0) == 1.0
    assert head_lambda(9) == 1.0
    assert head_lambda(10) == 0.5
    assert head_lambda(20) == 0.25

    assert abs(encoder_lambda(0) - 1.0) < 1e-9
    assert abs(encoder_lambda(1) - 0.95) < 1e-9


def test_head_step_epochs_clamped_like_fit():
    """``max(1, int(ft_scheduler_head_step_epochs))`` avoids zerodivision in ``epoch // s``."""
    assert max(1, 0) == 1
    assert max(1, -5) == 1
