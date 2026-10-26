import unittest
import numpy as np
from core import *
def unit(x):return np.stack([np.cos(x),np.sin(x)],-1).astype('float32')
class Tests(unittest.TestCase):
 def make(self,mode='full'):
  return Gate(unit(np.linspace(-.1,.1,16)),unit(np.array([[-.06,.03],[.02,.07]])),unit(np.array([[.15,.18],[.3,.34]])),unit(np.array([[.8,.85],[1.,1.05]])),np.ones((2,2),bool),.1,.1,mode,Config(capacity=4,candidate_k=2,min_witness=2,witness_capacity=4))
 def test_dominance(self):
  g=self.make('unconditional');q=unit(np.linspace(-2,2,100).reshape(50,2))
  for i in range(20):g.step(unit(np.array([i*.1,i*.1+.02])),str(i));self.assertTrue(np.all(scores(q,g.memory)<=scores(q,g.B)+1e-12))
 def test_shadow_immutable_capacity(self):
  g=self.make('unconditional');h=digest(g.B)
  for i in range(12):
   x=unit(np.array([i*.1,i*.1+.02]));expected=g.propose(x);r=g.step(x,str(i))
   if r['accepted']:np.testing.assert_array_equal(expected,g.S)
  self.assertEqual(h,digest(g.B));self.assertLessEqual(len(g.S),4)
 def test_predict_first(self):
  g=self.make('unconditional')
  for i in range(5):
   x=unit(np.array([i*.2,i*.2+.03]));p=float(scores(x,g.memory));r=g.step(x,str(i));self.assertEqual(p,r['score_before'])
 def test_identity(self):
  g=self.make();x=unit(np.array([.2,.3]));g.step(x,'a')
  with self.assertRaises(ValueError):g.step(x,'a')
 def test_label_interface(self):
  with self.assertRaises(TypeError):self.make().step(unit(np.array([.2,.3])),'a',label=1)
 def test_floor(self):
  g=self.make()
  for i in range(30):g.step(unit(np.array([i*.03,i*.03+.01])),str(i));self.assertTrue(np.all(g.margin(g.memory)>=g.cfg.kappa*g.m0-1e-9))
 def test_replay(self):
  a=self.make();b=self.make()
  for i in range(12):
   x=unit(np.array([i*.05,i*.05+.01]));self.assertEqual(a.step(x,str(i))['accepted'],b.step(x,str(i))['accepted']);np.testing.assert_array_equal(a.S,b.S)
 def test_proposals(self):
  a=self.make();b=self.make();a.S=unit(np.array([.48,.5]));b.S=unit(np.array([-.1,.05]));x=unit(np.array([.1,.5]));np.testing.assert_array_equal(a.propose(x)[-2:],b.propose(x)[-2:])
 def test_distance(self):np.testing.assert_allclose(distances(np.array([[0.,0.],[3.,4.]]),np.array([[0.,0.]])),[0.,5.])
 def test_safeguarded_operating_point(self):
  B=unit(np.array([0.,.1]));M=np.concatenate([B,unit(np.array([.5]))]);q=unit(np.linspace(-2,2,80).reshape(40,2));tau=.4
  source=scores(q,B);protected=safeguarded_scores(q,B,M,tau,.5)
  np.testing.assert_array_equal(source>tau,protected>tau)
  np.testing.assert_allclose(safeguarded_scores(q,B,M,tau,1.),source)
if __name__=='__main__':unittest.main(verbosity=2)
