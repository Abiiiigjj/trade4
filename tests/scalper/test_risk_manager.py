import pytest
from trade4.scalper.risk_manager import size_position, circuit_breaker_triggered, PositionParams

BALANCE = 2000.0


def test_normal_long_position():
    # entry=100, sl=98.5 → sl_distance=1.5, risk=2%=40 EUR, qty=40/1.5=26.67
    params = size_position(
        balance=BALANCE, entry_price=100.0, sl_price=98.5, side="long"
    )
    assert params.qty == pytest.approx(40.0 / 1.5, rel=1e-4)
    assert params.sl_price == 98.5
    assert params.liq_buffer_ok is True
    assert params.actual_risk_eur == pytest.approx(40.0, rel=1e-2)


def test_normal_short_position():
    params = size_position(
        balance=BALANCE, entry_price=100.0, sl_price=101.5, side="short"
    )
    assert params.qty == pytest.approx(40.0 / 1.5, rel=1e-4)
    assert params.sl_price == 101.5


def test_leverage_cap_reduces_qty():
    # entry=100, sl=99.99 → sl_distance=0.01, uncapped qty=40/0.01=4000
    # notional=400000, requires 200x leverage → capped at 20x
    # capped qty = (2000 * 20) / 100 = 400
    params = size_position(
        balance=BALANCE, entry_price=100.0, sl_price=99.99, side="long",
        max_risk_fraction=0.02, max_leverage=20,
    )
    assert params.leverage == 20
    assert params.qty == pytest.approx(400.0, rel=1e-4)
    assert params.actual_risk_eur < BALANCE * 0.02  # risk was reduced


def test_zero_sl_distance_raises():
    with pytest.raises(ValueError, match="sl_price must differ"):
        size_position(balance=BALANCE, entry_price=100.0, sl_price=100.0, side="long")


def test_liq_buffer_violated_when_leverage_too_high():
    # Use max_leverage=100 to force liquidation price very close to SL
    params = size_position(
        balance=BALANCE, entry_price=100.0, sl_price=99.0, side="long",
        max_leverage=100,
    )
    # With 100x leverage: liq_price ≈ 100*(1 - 1/100 + 0.005) = 100*0.995 = 99.5
    # SL=99.0 is BELOW liq_price=99.5 → buffer check should fail
    assert params.liq_buffer_ok is False


def test_circuit_breaker_triggered():
    assert circuit_breaker_triggered(
        realized_pnl_today=-61.0,
        day_start_balance=2000.0,
        daily_loss_limit=0.03,
    ) is True  # -61 < -60 (3% of 2000)


def test_circuit_breaker_not_triggered():
    assert circuit_breaker_triggered(
        realized_pnl_today=-59.0,
        day_start_balance=2000.0,
        daily_loss_limit=0.03,
    ) is False  # -59 > -60


def test_circuit_breaker_at_exact_limit():
    # Exactly at limit: not triggered (strict <)
    assert circuit_breaker_triggered(
        realized_pnl_today=-60.0,
        day_start_balance=2000.0,
        daily_loss_limit=0.03,
    ) is False
