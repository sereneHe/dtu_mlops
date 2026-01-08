from torch.utils.data import Dataset

from pic_classification_mnist_v01_xh.data import MyDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("data/raw")
    assert isinstance(dataset, Dataset)
