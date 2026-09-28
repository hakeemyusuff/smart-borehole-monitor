import unittest
import pandas as pd
from scripts.thesis_evidence import evidence_rows


class ThesisEvidenceTests(unittest.TestCase):
    def test_export_rejects_metrics_inconsistent_with_predictions(self):
        predictions=pd.DataFrame({'actual':[1.,2.], 'persistence':[1.,1.]})
        summary={'results':[{'key':'persistence','inputs':'level','model':'persistence',
                            'cv_mae_m':.2,'test_mae_m':.5,'test_rmse_m':2**-.5}]}
        rows=evidence_rows(summary,predictions)
        self.assertEqual(rows.evaluation_mae_cm.iloc[0],50.)
        predictions.loc[1,'persistence']=2.
        with self.assertRaises(ValueError):
            evidence_rows(summary,predictions)
