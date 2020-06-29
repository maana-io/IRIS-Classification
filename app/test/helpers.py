# coding: utf-8

import unittest
import pandas as pd

from pandas.util.testing import assert_frame_equal
from app.logic.helpers import (
    dataframeToDataset,
    datasetToDataframe,
    getDataFieldName,
    inferFeatureType, 
    id
)


class TestHelpers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.features = [
            {
                "feature": {"index": 0, "name": "feature 0", "type": "TEXT"},
                "data": [
                    {"text": "Hello" },
                    {"text": "Hello" }
                ]
            },
            {
                "feature": {"index": 1, "name": "feature 1", "type": "NUMERICAL"},
                "data": [
                    {"numerical": 1.2 },
                    {"numerical": 2.5 }
                ]
            },
            {
                "feature": {"index": 2, "name": "feature 2", "type": "SET"},
                "data": [
                    {"set": ["a", "b"]},
                    {"set": ["d", "e"]}
                ]
            }
        ]

        cls.ds = {
            "features": cls.features
        }

        cls.labeled_ds = {
            'id': id(),
            "data": {
                'id': id(),
                "features": cls.features
            },
            "label": {
                'id': id(),
                "feature": {'id': id(), "index": 0, "name": "label feature", "type": "LABEL"},
                "data": [
                    {'id': id(), "text": "value 1"},
                    {'id': id(), "text": "value 2"}
                ]
            }
        }

        cls.df_dict = {
            "feature 0": ["Hello", "Hello"],
            "feature 1": [1.2, 2.5],
            "feature 2": [
                ["a", "b"],
                ["d", "e"]
            ]
        }

        cls.labeled_df_dict = {
            **cls.df_dict,
            "label feature": [
                "value 1",
                "value 2"
            ],
        }

    
    def test_get_data_field_name(self):
        self.assertEqual('numerical', getDataFieldName('NUMERICAL'))
        self.assertEqual('text', getDataFieldName('CATEGORICAL'))
        self.assertEqual('text', getDataFieldName('TEXT'))
        self.assertEqual('set', getDataFieldName('SET'))
        self.assertEqual('numerical', getDataFieldName('BOOLEAN'))
    

    def test_infer_feature_type(self):
        numerical_series_1 = pd.Series([1,2,3])
        numerical_series_2 = pd.Series([0.57, 0.61])
        boolean_series_1 = pd.Series(['true','true'])
        boolean_series_2 = pd.Series(['false', 'false'])
        set_series_1 = pd.Series([[1,2], [3,4]])
        set_series_2 = pd.Series([(1,2), (3,4)])
        set_series_3 = pd.Series([{1,2}, {3,4}])
        text_series_1 = pd.Series(['Lorem ipsum dolor sit amet','consectetur adipiscing elit'])
        text_series_2 = pd.Series(['set_field::m=0.73;set_field::t=-0.28','cat_field::Health Care=0.68;set_field::i=-0.3'])

        self.assertEqual('NUMERICAL', inferFeatureType(numerical_series_1))
        self.assertEqual('NUMERICAL', inferFeatureType(numerical_series_2))
        self.assertEqual('BOOLEAN', inferFeatureType(boolean_series_1))
        self.assertEqual('BOOLEAN', inferFeatureType(boolean_series_2))
        self.assertEqual('SET', inferFeatureType(set_series_1))
        self.assertEqual('SET', inferFeatureType(set_series_2))
        self.assertEqual('SET', inferFeatureType(set_series_3))
        self.assertEqual('TEXT', inferFeatureType(text_series_1))
        self.assertEqual('TEXT', inferFeatureType(text_series_2))


    def test_dataframe_to_dataset(self):
        df = pd.DataFrame(self.df_dict)
        result = dataframeToDataset(df)
        expected = self.ds

        #generated IDs can't be the same
        result.pop('id')
        for f in result['features']:
            f.pop('id')
            f['feature'].pop('id')
            for ditem in f['data']:
                ditem.pop('id')

        self.assertEqual(expected, result)


    def test_dataset_to_dataframe(self):
        expected = pd.DataFrame(self.df_dict)
        result = datasetToDataframe(self.ds)
        assert_frame_equal(expected.reset_index(drop=True), result.reset_index(drop=True))


    def test_labeled_dataset_to_dataframe(self):
        expected = pd.DataFrame(self.labeled_df_dict)
        result = datasetToDataframe(self.labeled_ds)
        assert_frame_equal(expected.reset_index(drop=True), result.reset_index(drop=True))



if __name__ == '__main__':
    unittest.main()
