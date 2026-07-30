import numpy as np
import pyqtgraph.opengl as gl

import colors


class OpticalAxis:
    def __init__(self, eyeball_center, optical_axis_vector, color=(0.8, 0.2, 0.2, 1.0)):
        self.eyeball_center = eyeball_center
        self.optical_axis_vector = optical_axis_vector

        self.eyeball_rotation = (0.0, 0.0)  # (azimuth, elevation) in radians
        self.color = color

        self._create_mesh()

    def add_to_view(self, view):
        view.addItem(self.shaft)
        view.addItem(self.head)

    def update(self, new_eyeball_center, new_eyeball_rotation, new_optical_axis_vector):
        self.eyeball_center = new_eyeball_center
        self.optical_axis_vector = new_optical_axis_vector
        self.eyeball_rotation = new_eyeball_rotation

        self.shaft.resetTransform()
        self.head.resetTransform()

        self.shaft.scale(0.5, 0.5, 25.0)
        self.head.scale(0.5, 0.5, 3.0)

        self.shaft.rotate(-90, 1, 0, 0)
        self.shaft.translate(0, -self.shaft_length * 25, 0)
        self.shaft.rotate(self.eyeball_rotation[0] * 180 / np.pi, 0, 0, 1)
        self.shaft.rotate(self.eyeball_rotation[1] * 180 / np.pi, 1, 0, 0)

        self.head.rotate(-90, 1, 0, 0)
        self.head.translate(0, -self.shaft_length * 25, 0)
        self.head.rotate(self.eyeball_rotation[0] * 180 / np.pi, 0, 0, 1)
        self.head.rotate(self.eyeball_rotation[1] * 180 / np.pi, 1, 0, 0)

        self.shaft.translate(*self.eyeball_center)
        self.head.translate(*self.eyeball_center)

    def _create_mesh(self):
        shaft_radius = 0.5
        head_radius = 1.0
        head_length = 1.8

        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.0, 1.0, 0.0])
        vec = end - start
        length = np.linalg.norm(vec)
        if length == 0:
            raise ValueError("start and end cannot be the same point")
        self.direction = vec / length

        # Shaft length (everything except arrowhead)
        self.shaft_length = length - head_length

        # Shaft mesh
        shaft_mesh_data = gl.MeshData.cylinder(
            rows=10,
            cols=20,
            radius=[shaft_radius, shaft_radius],
            length=self.shaft_length,
        )
        self.shaft = gl.GLMeshItem(
            meshdata=shaft_mesh_data,
            smooth=True,
            color=self.color,
        )

        # Head mesh (tapered cylinder = cone)
        head_mesh_data = gl.MeshData.cylinder(
            rows=10, cols=20, radius=[head_radius, 0.0], length=head_length
        )
        self.head = gl.GLMeshItem(
            meshdata=head_mesh_data,
            smooth=True,
            color=self.color,
        )


class Disk:
    def __init__(
        self,
        eyeball_center,
        eyeball_radius,
        disk_radius,
        color=(0.8, 0.8, 0.8, 1.0),
        is_pupil=False,
    ):
        self.eyeball_center = eyeball_center
        self.eyeball_radius = eyeball_radius

        self.disk_radius = disk_radius
        self.color = color
        self.is_pupil = is_pupil

        self.position = self.eyeball_center + 0.88 * self.eyeball_radius * np.array(
            [0, 1, 0]
        )
        if self.is_pupil:
            self.position += np.array([0, 0.05, 0.0])

        self._create_mesh()

    def add_to_view(self, view):
        view.addItem(self.mesh_item)

    def update(
        self,
        new_eyeball_center,
        new_eyeball_rotation,
        new_disk_radius,
    ):
        self.eyeball_center = new_eyeball_center
        self.eyeball_rotation = new_eyeball_rotation
        self.disk_radius = new_disk_radius

        self.position = 0.88 * self.eyeball_radius * np.array([0, 1, 0])
        if self.is_pupil:
            self.position += np.array([0, 0.05, 0.0])

        self.mesh_item.resetTransform()
        self.mesh_item.scale(self.disk_radius, self.disk_radius, self.disk_radius)
        self.mesh_item.translate(*self.position)
        self.mesh_item.rotate(self.eyeball_rotation[0] * 180 / np.pi, 0, 0, 1)
        self.mesh_item.rotate(self.eyeball_rotation[1] * 180 / np.pi, 1, 0, 0)

        self.mesh_item.translate(*self.eyeball_center)

    def _create_mesh(self):
        resolution = 64

        # Angles around the circle
        theta = np.linspace(0, 2 * np.pi, resolution, endpoint=False)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        verts = []
        # Outer ring
        for i in range(resolution):
            verts.append([cos_t[i], 0.0, sin_t[i]])
        # Center vertex
        verts.append([0.0, 0.0, 0.0])

        verts = np.array(verts, dtype=float)

        faces = []
        # Fan triangles from center
        center_idx = resolution
        for i in range(resolution):
            j = (i + 1) % resolution
            faces.append([center_idx, i, j])

        faces = np.array(faces, dtype=int)

        mesh_data = gl.MeshData(vertexes=verts, faces=faces)
        self.mesh_item = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=True,
            color=self.color,
        )
        self.mesh_item.setGLOptions("opaque")


class EyeLid:
    def __init__(self, eyeball_center, eyeball_radius, color=(1.0, 1.0, 1.0, 1.0)):
        self.eyeball_center = eyeball_center
        self.eyeball_radius = eyeball_radius
        self.color = color

        self.position = self.eyeball_center
        self.eyelid_angle = 0.0  # Radians

        self.eyelid_radius = self.eyeball_radius * 1.2

        self._create_mesh()

    def add_to_view(self, view):
        view.addItem(self.mesh_item)

    def update(self, new_eyeball_center, new_eyelid_angle):
        self.eyeball_center = new_eyeball_center
        self.eyelid_angle = new_eyelid_angle

        self.position = self.eyeball_center

        self.mesh_item.resetTransform()
        self.mesh_item.rotate(self.eyelid_angle * 180 / np.pi, 1, 0, 0)
        self.mesh_item.translate(*self.position)

    def _create_mesh(self):
        tube_radius = 0.09
        n_curve = 20
        n_circle = 12

        # Curve parameter (arc around circle)
        theta = np.linspace(np.pi / 8, np.pi - np.pi / 8, n_curve)
        R = self.eyelid_radius
        r = tube_radius

        # Cross-section circle
        phi = np.linspace(0, 2 * np.pi, n_circle, endpoint=False)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # Storage for vertices
        verts = []

        for i, t in enumerate(theta):
            # Center of tube along arc
            cx = R * np.cos(t)
            cy = R * np.sin(t)
            cz = 0

            # Tangent vector along arc
            tangent = np.array([-R * np.sin(t), R * np.cos(t), 0.0])
            tangent /= np.linalg.norm(tangent)

            # Normal (Z-axis)
            normal = np.array([0, 0, 1.0])
            binormal = np.cross(tangent, normal)

            # Cross-section vertices
            for j in range(n_circle):
                x = cx + r * (cos_phi[j] * normal[0] + sin_phi[j] * binormal[0])
                y = cy + r * (cos_phi[j] * normal[1] + sin_phi[j] * binormal[1])
                z = cz + r * (cos_phi[j] * normal[2] + sin_phi[j] * binormal[2])
                verts.append([x, y, z])

        verts = np.array(verts, dtype=float)

        # Build faces (two triangles per quad)
        faces = []
        for i in range(n_curve - 1):
            for j in range(n_circle):
                # Index of this and next circle
                i0 = i * n_circle + j
                i1 = i * n_circle + (j + 1) % n_circle
                i2 = (i + 1) * n_circle + j
                i3 = (i + 1) * n_circle + (j + 1) % n_circle

                # Two triangles for this quad
                faces.append([i0, i2, i1])
                faces.append([i1, i2, i3])

        faces = np.array(faces, dtype=int)

        mesh_data = gl.MeshData(vertexes=verts, faces=faces)
        self.mesh_item = gl.GLMeshItem(
            meshdata=mesh_data, smooth=True, color=self.color
        )
        self.mesh_item.setGLOptions("opaque")


class Eyeball:
    def __init__(self, eyeball_center, eyeball_radius=12.0, color=(0.5, 0.5, 0.5, 1.0)):
        self.eyeball_center = eyeball_center
        self.eyeball_radius = eyeball_radius
        self.color = color

        self.position = self.eyeball_center

        # These will be computed when the `update()` method is called.
        self.optical_axis_vector = None
        self.eyeball_rotation = None

        self._create_mesh()

    def add_to_view(self, view):
        view.addItem(self.line_item)
        view.addItem(self.mesh_item)

    def update(self, new_eyeball_center, new_optical_axis_vector):
        self.eyeball_center = new_eyeball_center
        self.optical_axis_vector = new_optical_axis_vector

        self._compute_eyeball_rotation()

        self.position = self.eyeball_center

        self.mesh_item.resetTransform()
        self.mesh_item.rotate(self.eyeball_rotation[0] * 180 / np.pi, 0, 0, 1)
        self.mesh_item.rotate(self.eyeball_rotation[1] * 180 / np.pi, 1, 0, 0)
        self.mesh_item.translate(*self.position)

        self.line_item.resetTransform()
        self.line_item.rotate(self.eyeball_rotation[0] * 180 / np.pi, 0, 0, 1)
        self.line_item.rotate(self.eyeball_rotation[1] * 180 / np.pi, 1, 0, 0)
        self.line_item.translate(*self.position)

    def _compute_eyeball_rotation(self):
        axis_x, axis_y, axis_z = self.optical_axis_vector
        optical_axis_norm = np.linalg.norm(self.optical_axis_vector)

        azimuth = np.arctan2(axis_y, axis_x)  # Rotation about Z axis
        elevation = np.arcsin(axis_z / optical_axis_norm)  # Rotation about Y axis

        azimuth -= np.pi / 2  # 0 azimuth is forward for Neon

        self.eyeball_rotation = (azimuth, elevation)

    def _create_mesh(self):
        self._create_black_body()
        self._create_wireframe()

    def _create_black_body(
        self,
        theta_min=np.pi / 6,
        theta_max=np.pi,
        phi_steps=32,
        theta_steps=32,
    ):
        # Polar angle θ ∈ [theta_min, theta_max]
        theta = np.linspace(theta_min, theta_max, theta_steps)
        # Azimuthal angle φ ∈ [0, 2π)
        phi = np.linspace(0, 2 * np.pi, phi_steps, endpoint=False)

        verts = []
        for t in theta:
            for p in phi:
                x = self.eyeball_radius * np.sin(t) * np.cos(p)
                y = self.eyeball_radius * np.cos(t)
                z = self.eyeball_radius * np.sin(t) * np.sin(p)
                verts.append([x, y, z])
        verts = np.array(verts, dtype=float)

        faces = []
        for i in range(theta_steps - 1):
            for j in range(phi_steps):
                # indices in the grid
                i0 = i * phi_steps + j
                i1 = i * phi_steps + (j + 1) % phi_steps
                i2 = (i + 1) * phi_steps + j
                i3 = (i + 1) * phi_steps + (j + 1) % phi_steps

                # two triangles per quad (consistent CCW winding)
                faces.append([i0, i2, i1])
                faces.append([i1, i2, i3])

        faces = np.array(faces, dtype=int)

        mesh_data = gl.MeshData(vertexes=verts, faces=faces)
        self.mesh_item = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=True,
            color=np.array(colors.BACKGROUND) / 255,
            shader="shaded",
        )
        self.mesh_item.setGLOptions("opaque")

    def _create_wireframe(
        self,
        theta_min=np.pi / 6,
        theta_max=np.pi,
        phi_steps=32,
        theta_steps=10,
    ):
        # Polar angle θ ∈ [theta_min, theta_max]
        theta = np.linspace(theta_min, theta_max, theta_steps)

        rad = self.eyeball_radius * 1.015

        x, y, z = [], [], []
        for t in theta:
            for p in np.linspace(0, 2 * np.pi, 200):
                x.append(rad * np.sin(t) * np.cos(p))
                y.append(rad * np.cos(t))
                z.append(rad * np.sin(t) * np.sin(p))

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        # Collect the coordinates into a single array.
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

        # Azimuthal angle φ ∈ [0, 2π)
        phi = np.linspace(0, 2 * np.pi, phi_steps)

        for p in phi:
            for t in np.linspace(theta_min, theta_max, 200):
                x = rad * np.sin(t) * np.cos(p)
                y = rad * np.cos(t)
                z = rad * np.sin(t) * np.sin(p)
                pts = np.vstack([pts, [x, y, z]])

        for p_start in phi:
            p = p_start
            for t in np.linspace(theta_min, theta_max, 200):
                x = rad * np.sin(t) * np.cos(p)
                y = rad * np.cos(t)
                z = rad * np.sin(t) * np.sin(p)
                pts = np.vstack([pts, [x, y, z]])

                p += np.pi / 420

        self.line_item = gl.GLLinePlotItem(
            pos=pts, width=2.5, mode="line_strip", color=self.color
        )
        self.line_item.setGLOptions("opaque")


class ThreeDEyeModel:
    def __init__(self, eyeball_center, eyeball_radius=12, color=None):
        self.eyeball_center = eyeball_center
        self.eyeball_radius = eyeball_radius
        self.color = color

        self.eyeball_center = np.array([0.0, 0.0, 0.0])
        self.pupil_diameter = 4.0
        self.eyelid_top_angle = 0.2
        self.eyelid_bottom_angle = -0.2
        self.optical_axis_vector = np.array([0.0, 1.0, 0.0])

        self.eyeball = Eyeball(
            eyeball_center=self.eyeball_center,
            eyeball_radius=self.eyeball_radius,
            color=colors.EYEBALL_GRAY,
        )
        self.optical_axis = OpticalAxis(
            eyeball_center=self.eyeball_center,
            optical_axis_vector=self.optical_axis_vector,
            color=self.color,
        )
        self.top_eye_lid = EyeLid(
            eyeball_center=self.eyeball_center,
            eyeball_radius=self.eyeball_radius,
            color=colors.EYELID_GRAY,
        )
        self.bottom_eye_lid = EyeLid(
            eyeball_center=self.eyeball_center,
            eyeball_radius=self.eyeball_radius,
            color=colors.EYELID_GRAY,
        )
        self.iris = Disk(
            eyeball_center=self.eyeball_center,
            eyeball_radius=self.eyeball_radius,
            disk_radius=6.1,
            color=colors.IRIS_GRAY,
        )
        self.pupil = Disk(
            eyeball_center=self.eyeball_center + np.array([0, 0.1, 0.0]),
            eyeball_radius=self.eyeball_radius,
            disk_radius=self.pupil_diameter / 2.0,
            color=colors.PUPIL_GRAY,
            is_pupil=True,
        )

    def add_to_view(self, view):
        self.eyeball.add_to_view(view)
        self.optical_axis.add_to_view(view)
        self.top_eye_lid.add_to_view(view)
        self.bottom_eye_lid.add_to_view(view)
        self.iris.add_to_view(view)
        self.pupil.add_to_view(view)

    def update(
        self,
        new_eyeball_center,
        new_optical_axis_vector,
        new_pupil_diameter,
        new_eyelid_top_angle,
        new_eyelid_bottom_angle,
    ):
        self.eyeball_center = np.array(
            [
                new_eyeball_center[0],
                new_eyeball_center[2],
                -1 * new_eyeball_center[1],
            ]
        )

        self.optical_axis_vector = np.array(
            [
                new_optical_axis_vector[0],
                new_optical_axis_vector[2],
                -1 * new_optical_axis_vector[1],
            ]
        )

        self.pupil_diameter = new_pupil_diameter

        self.eyelid_top_angle = new_eyelid_top_angle
        self.eyelid_bottom_angle = new_eyelid_bottom_angle

        self.eyeball.update(self.eyeball_center, self.optical_axis_vector)
        self.optical_axis.update(
            self.eyeball_center, self.eyeball.eyeball_rotation, self.optical_axis_vector
        )
        self.iris.update(self.eyeball_center, self.eyeball.eyeball_rotation, 6.1)
        self.pupil.update(
            self.eyeball_center,
            self.eyeball.eyeball_rotation,
            self.pupil_diameter / 2.0,
        )
        self.top_eye_lid.update(self.eyeball_center, self.eyelid_top_angle)
        self.bottom_eye_lid.update(self.eyeball_center, self.eyelid_bottom_angle)
