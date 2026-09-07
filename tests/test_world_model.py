import unittest
import numpy as np
from dataclasses import replace

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from lm_testbed.world import MachineConfig, DiagnosticConfig, simulate, summaries, observe, fingerprint
    from lm_testbed.learning import grid_posterior
    from lm_testbed.language import TinyToolLM, batch, partitions, parse_call, TOKEN
    from lm_testbed.runtime import dispatch, run_agent

@unittest.skipUnless(torch is not None, "Optional PyTorch dependency not installed")
class WorldContracts(unittest.TestCase):
    def test_zero_coupling_limit_approach(self):
        cfg=replace(MachineConfig(),coupling=1e-10)
        _,s=simulate(cfg=cfg)
        self.assertLess(float((s[:,:,2]-1).abs().max()),1e-8)
        self.assertGreater(float(s[:,:,1].max()),.1)
    def test_more_mass_reduces_inward_motion(self):
        p=torch.tensor([[1.,.7,.22,.6],[1.,1.3,.22,.6]])
        _,s=simulate(p)
        self.assertGreater(float(s[1,-1,2]),float(s[0,-1,2]))
    def test_same_config_and_seed_reproduce_raw_data(self):
        t,s=simulate(); a=observe(t,s[0],seed=55); b=observe(t,s[0],seed=55)
        np.testing.assert_array_equal(a['diagnostics']['pdv']['voltage'],b['diagnostics']['pdv']['voltage'])
    def test_missing_diagnostic_is_absent(self):
        t,s=simulate(); record=observe(t,s[0],DiagnosticConfig(missing=('pdv','image')))
        self.assertNotIn('pdv',record['diagnostics']); self.assertNotIn('image',record['diagnostics'])
    def test_summary_operator_cannot_observe_growth(self):
        a=torch.tensor([[1.,1.,.22,.3],[1.,1.,.22,.9]])
        torch.testing.assert_close(summaries(a)[0],summaries(a)[1])
    def test_configuration_validation(self):
        for kw in ({'mass':0},{'charge':float('nan')},{'samples':12},{'end_time':2}):
            with self.assertRaises(ValueError): MachineConfig(**kw)
        self.assertNotEqual(fingerprint(MachineConfig()),fingerprint(replace(MachineConfig(),mass=1.1)))
    def test_grid_recovers_noiseless_on_grid_parameters(self):
        target=torch.tensor([[1.,1.,.22,.6]])
        with torch.no_grad(): obs=summaries(target)[0]
        p=grid_posterior(obs,n=21)
        maximum=p['theta'][p['weights'].argmax()]
        torch.testing.assert_close(maximum[:2],target[0,:2])
        self.assertAlmostEqual(float(p['weights'].sum()),1.,places=5)
    def test_causal_mask_blocks_future_tokens(self):
        model=TinyToolLM().eval(); x,_=batch(partitions()[0]); changed=x[:1].clone(); changed[0,-1]=TOKEN['missing']
        with torch.no_grad(): a=model(x[:1]); b=model(changed)
        torch.testing.assert_close(a[:,:-1],b[:,:-1])
    def test_disjoint_semantic_partitions(self):
        groups=[{tuple(sorted(x.items())) for x in rows} for rows in partitions()]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        self.assertEqual(sum(map(len,groups)),24)
    def test_missing_summary_never_becomes_estimate(self):
        record={'case':{'question':'mass','current':False,'motion':False,'clock':False},'summaries':torch.full((12,),float('nan'))}
        result=run_agent(lambda c:{'tool':'infer'},record)
        self.assertEqual(result['final']['status'],'unresolved')
    def test_invalid_and_extra_tool_keys_rejected(self):
        for raw in ['not json','{"tool":"shell"}','{"tool":"infer","truth":1}']:
            self.assertIsNone(parse_call(raw))
    def test_all_required_data_can_produce_estimate(self):
        obs=summaries(torch.tensor([[1.,1.,.22,.6]]))[0].detach()
        record={'case':{'question':'mass','current':True,'motion':True,'clock':True},'summaries':obs}
        result=dispatch({'tool':'infer'},record)
        self.assertEqual(result['status'],'conditional_estimate')
        self.assertLessEqual(result['interval90'][0],1.)
        self.assertGreaterEqual(result['interval90'][1],1.)

if __name__=='__main__': unittest.main()

class RunnerScope(unittest.TestCase):
    def test_runner_selects_only_new_course_notebooks(self):
        import tempfile
        from pathlib import Path
        try:
            import nbformat
        except ModuleNotFoundError:
            self.skipTest("Optional notebook dependency not installed")
        from scripts.execute_notebooks import course_notebooks
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'notebooks').mkdir()
            for number,tag in [('01','01'),('04','04'),('05','05'),('06',None)]:
                notebook=nbformat.v4.new_notebook()
                if tag: notebook.metadata['lm_testbed']={'lesson':tag}
                nbformat.write(notebook,root/'notebooks'/f'{number}_example.ipynb')
            self.assertEqual([p.name for p in course_notebooks(root)],['05_example.ipynb'])
