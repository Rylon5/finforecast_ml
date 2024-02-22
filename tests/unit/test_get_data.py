import pytest

from app import main


def test_get_data_not_empty():
    ticker = ['IBM', 'NVR', 'AZO', 'AVGO', 'ADBE', 'MSFT', 'GOOG', 'TEAM', 'HLT', 'AMZN', 'GRMN', 'OTTR', 'TEX', 'CSCO',
              'BEPC', 'PSO', 'INFN', 'KODK', 'KRON', 'AMBO', 'WF', 'OBK', 'VVV', 'VET', 'DDD', 'CTRA', 'RRR', 'CSIQ']
    assert not main.get_data(ticker_symbols=ticker).empty


if __name__ == "__main__":
    pytest.main()
