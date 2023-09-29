from financegpt.data.dataset import Dataset


def test_text_dataset_creation(text_data_point):
    text_dataset = Dataset(data=[text_data_point])
    assert len(text_dataset) == 1
    assert text_dataset[0] == text_data_point


def test_ohlc_dataset_creation(ohlc_data_point):
    ohlc_dataset = Dataset(data=[ohlc_data_point])
    assert len(ohlc_dataset) == 1
    assert ohlc_dataset[0] == ohlc_data_point
