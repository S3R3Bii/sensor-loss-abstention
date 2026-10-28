from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare import mask,FEATURES,weights
from analyze import selective

class Contracts(unittest.TestCase):
    def frame(self):return pd.DataFrame([{**{c:1. for c in FEATURES},'participant':1,'label':0}])
    def test_accelerometer_loss_removes_both_derived_streams(self):
        m=mask(self.frame(),'acc_missing')
        self.assertTrue((m.filter(regex='^(body_acc|total_acc)').to_numpy()==0).all())
        self.assertTrue((m.filter(regex='^body_gyro').to_numpy()==1).all())
        self.assertEqual(m.acc_present.iloc[0],0)
    def test_gyroscope_loss_retains_acceleration(self):
        m=mask(self.frame(),'gyro_missing')
        self.assertTrue((m.filter(regex='^body_gyro').to_numpy()==0).all())
        self.assertTrue((m.filter(regex='^(body_acc|total_acc)').to_numpy()==1).all())
    def test_absent_inputs_force_abstention_even_with_confident_model(self):
        m=mask(self.frame(),'both_missing')
        for i in range(6):m[f'p{i}']=1. if i==0 else 0.
        self.assertEqual(selective(m,0.)['accepted'],0)
    def test_participant_training_weights_have_equal_totals(self):
        f=pd.DataFrame({'participant':[1,2,2,2]});w=weights(f)
        self.assertAlmostEqual(w[0],sum(w[1:]));self.assertAlmostEqual(sum(w),4.)
    def test_threshold_boundary_is_inclusive(self):
        f=mask(self.frame(),'intact')
        for i in range(6):f[f'p{i}']=.5 if i in [0,1] else 0.
        self.assertEqual(selective(f,.5)['accepted'],1)
        self.assertEqual(selective(f,.50001)['accepted'],0)

if __name__=='__main__':unittest.main()
