"""데이터 입력 어댑터 (7~8주차 구현 예정 — 인터페이스만 고정).

- load_card_csv: 카드·은행 거래내역 → 월별 지출 시계열
- KISAdapter: 한국투자증권 Open API로 본인 계좌 잔고·보유종목 조회
"""
from dataclasses import dataclass
from typing import Dict, List


def load_card_csv(path: str) -> Dict[str, float]:
    """반환: {'YYYY-MM': 지출합계(만원)}"""
    raise NotImplementedError("7주차: 카드사별 CSV 컬럼 매핑 후 구현")


@dataclass
class Holding:
    ticker: str
    qty: float
    price: float


class KISAdapter:
    def __init__(self, app_key: str, app_secret: str, account: str):
        self.app_key, self.app_secret, self.account = app_key, app_secret, account

    def holdings(self) -> List[Holding]:
        raise NotImplementedError("7주차: KIS 잔고조회 API 연동")

    def total_value(self) -> float:
        return sum(h.qty * h.price for h in self.holdings())
