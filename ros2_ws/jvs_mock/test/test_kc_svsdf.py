"""Numerical contract checks, not a substitute for the Isaac acceptance suite."""
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from jvs_mock.kc_svsdf import SweptCollision, advance, primitive_sdf
from jvs_mock.whole_body_qp import whole_body_kinematics


class CollisionContract(unittest.TestCase):
    def test_groundtruth_mesh_cache_selects_local_surfels(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'mesh.npz'
            centers=np.array([[.4,0,.2],[.45,.1,.3],[2.,2.,1.]])
            metadata=dict(format='jvs_groundtruth_mesh/v1')
            np.savez(path,vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),
                     faces=np.array([[0,1,2]]),surfel_centers=centers,
                     surfel_radius=.04,minimum_clearance=.025,
                     metadata=np.asarray(json.dumps(metadata)))
            collision=SweptCollision(path)
            worker_copy=SweptCollision(collision.scene)
            active,_=collision.active_obstacles(np.zeros(9))
            self.assertLess(len(active),len(centers))
            self.assertTrue(np.all(np.isfinite(collision.clearance(np.zeros(9)))))
            self.assertTrue(worker_copy.is_mesh)

    def collision(self):
        return SweptCollision(dict(format='jvs_obstacles/v1',minimum_clearance_m=.025,
            obstacles=[dict(type='sphere',name='test',center=[.27,.09,.62],radius=.06)]))

    def test_all_body_distance_gradients(self):
        c=self.collision();x=np.array([.01,-.02,.3,.7,-.3,.6,.2,-.5,.1]);eps=1e-6
        _,g=c.clearance(x,gradient=True)
        fd=np.stack([(c.clearance(x+np.eye(9)[i]*eps)-c.clearance(x-np.eye(9)[i]*eps))/(2*eps) for i in range(9)],axis=-1)
        np.testing.assert_allclose(g,fd,atol=2e-6)
        frames,_,_=c.geometry.frames(x)
        p,r,_=whole_body_kinematics(x)
        np.testing.assert_allclose(frames[7][0],p,atol=1e-12)
        np.testing.assert_allclose(frames[7][1],r,atol=1e-12)

    def test_sweep_encloses_dense_reference(self):
        c=self.collision();x=np.array([0,0,.2,.7,-.3,.6,.2,-.5,.1])
        u=np.array([.12,.3,.1,-.1,.1,.1,-.1,.1]);duration=.6
        result=c.sweep(x,u,duration,tolerance=.002)
        dense=np.min([c.clearance(advance(x,u,t)) for t in np.linspace(0,duration,601)],axis=0)
        self.assertTrue(np.all(result['lower']<=dense+1e-8))
        self.assertTrue(np.all(result['upper']>=dense-1e-4))
        self.assertLessEqual(result['error'],.002)

    def test_all_primitive_gradients(self):
        c=self.collision();x=np.array([.01,-.02,.3,.7,-.3,.6,.2,-.5,.1]);eps=1e-6
        d,g=c.geometry.primitive_distances(x,c.centers)
        fd=np.stack([(c.geometry.primitive_distances(x+np.eye(9)[i]*eps,c.centers)[0]-
                      c.geometry.primitive_distances(x-np.eye(9)[i]*eps,c.centers)[0])/(2*eps)
                     for i in range(9)],axis=-1)
        np.testing.assert_allclose(g,fd,atol=2e-6)
        self.assertEqual(d.shape[1],len(c.geometry.shapes))

    def test_hidden_midpoint_collision(self):
        c=self.collision();c.centers=np.array([[0,0,.13]]);c.radii=np.array([.02])
        x=np.zeros(9);x[0]=-.4;u=np.zeros(8);u[0]=.8
        self.assertGreater(c.clearance(x).min(),0)
        self.assertGreater(c.clearance(advance(x,u,1)).min(),0)
        self.assertLess(c.sweep(x,u,1)['lower'].min(),0)

    def test_sweep_retains_measured_base_tilt(self):
        c=self.collision();x=np.array([0,0,.2,.7,-.3,.6,.2,-.5,.1])
        br=Rotation.from_euler('xyz',[.015,-.02,x[2]]).as_matrix()
        expected=c.clearance(x,.003,br)
        result=c.sweep(x,np.zeros(8),.02,z=.003,base_rotation=br)
        np.testing.assert_allclose(result['lower'],expected,atol=1e-12)
        self.assertGreater(np.max(np.abs(expected-c.clearance(x,.003))),1e-4)

    def test_filtered_interior_direction_consistency(self):
        c=self.collision();x=np.array([0,0,.2,.7,-.3,.6,.2,-.5,.1])
        frames,_,_=c.geometry.frames(x);point=frames[4][0]
        accumulator=np.arange(9,dtype=float)-4.5
        result=c.interior([x],0.,point,4,.08,accumulator,True)
        self.assertIsNotNone(result)
        d,g=result
        self.assertLess(d,0)
        self.assertTrue(np.all(g[[4,5]]*accumulator[[4,5]]>=-1e-12))

    def test_primitive_inside_outside(self):
        self.assertAlmostEqual(primitive_sdf([[0,0,0]],'box',[2,2,2])[0],-1.)
        self.assertAlmostEqual(primitive_sdf([[0,0,0]],'cylinder',[1,2])[0],-1.)
        self.assertAlmostEqual(primitive_sdf([[0,0,0]],'frustum',[.18,.13,.12])[0],-.06)

    def test_analytic_normals_against_independent_distance_differences(self):
        rng = np.random.default_rng(83)
        points = rng.uniform(-.35, .35, (200, 3))
        for kind, size in [('sphere', [.08]), ('box', [.2, .3, .15]),
                           ('cylinder', [.12, .25]), ('frustum', [.18, .13, .12])]:
            distance, normal = primitive_sdf(points, kind, size, with_gradient=True)
            eps = 1e-7
            independent = np.column_stack([
                (primitive_sdf(points+eps*np.eye(3)[i], kind, size)
                 -primitive_sdf(points-eps*np.eye(3)[i], kind, size))/(2*eps)
                for i in range(3)])
            np.testing.assert_allclose(normal, independent, atol=2e-7)
            np.testing.assert_array_equal(distance, primitive_sdf(points, kind, size))
            np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1., atol=1e-12)

    def test_normals_are_finite_at_primitive_axis_and_surface(self):
        for kind, size in [('sphere', [.08]), ('box', [.2, .3, .15]),
                           ('cylinder', [.12, .25]), ('frustum', [.18, .13, .12])]:
            _, normal = primitive_sdf(np.array([[0., 0., 0.], [0., 0., .06],
                                              [.13, 0., .06]]), kind, size, with_gradient=True)
            self.assertTrue(np.all(np.isfinite(normal)))
            self.assertTrue(np.all(np.linalg.norm(normal, axis=1) <= 1.+1e-12))


if __name__=='__main__':unittest.main()
