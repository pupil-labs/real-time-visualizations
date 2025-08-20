import numpy as np
from scipy.spatial.transform import Rotation as R

# We recommend reviewing the relevant datastreams in Neon's Documentation, if necessary,
# before working your way through this code:
# https://docs.pupil-labs.com/neon/data-collection/data-streams/

# Our ThreeDEyeModel class, defined at the bottom of this file, is essentially
# a container for the 3D representation of an eye, including:
# - A sphere for the main eyeball body.
# - A disk for the pupil.
# - An arrow for the optical axis vector.
# - A curved tube for each eyelid.
# These components are separated into their own classes to make it easier to
# manage their individual behaviors and properties, as well as to understand
# how they combine to form a model of the eye.


class OpticalAxis:
    """
    Represents the optical axis of the eye, defined by the eye's estimated center and optical axis vector.

    Attributes:
        eyeball_center (np.ndarray): The center point of the eyeball.
        optical_axis_vector (np.ndarray): The direction vector of the optical axis.
        optical_axis_quiver_plot (matplotlib.quiver.Quiver): The quiver plot representing the optical axis.
    """

    def __init__(self, eyeball_center, optical_axis_vector):
        self.eyeball_center = eyeball_center
        self.optical_axis_vector = optical_axis_vector

        # This will be computed when the `update()` method is called.
        self.optical_axis_quiver_plot = None

    def update(self, eyeball_center, optical_axis_vector):
        """
        Update the optical axis representation based on new eyeball center and optical axis estimates from Neon.
        """

        self.eyeball_center = eyeball_center
        self.optical_axis_vector = optical_axis_vector

    def plot(self, ax, color):
        """
        Plots the optical axis as an arrow on the given axis.

        Args:
            ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis to plot on.
            color (str or tuple): Color for the optical axis arrow.
        """

        if self.optical_axis_quiver_plot in ax.artists:
            self.optical_axis_quiver_plot.remove()

        self.optical_axis_quiver_plot = ax.quiver(
            self.eyeball_center[0],
            self.eyeball_center[1],
            self.eyeball_center[2],
            *self.optical_axis_vector,
            color=color,
            length=30.0,
            normalize=True,
        )


class EyeBall:
    """
    Represents the eyeball's body as a sphere with an opening to simulate the iris.

    Attributes:
        eyeball_center (np.ndarray): The center point of the eyeball.
        eyeball_radius (float): The radius of the eyeball.
        optical_axis_vector (np.ndarray): The direction vector of the optical axis.
        eyeball_rotation (scipy.spatial.transform.Rotation): The rotation of the eyeball, as derived from the optical axis.
    """

    def __init__(self, eyeball_center, eyeball_radius=12):
        self.eyeball_center = eyeball_center
        self.eyeball_radius = eyeball_radius

        # These will be computed when the `update()` method is called.
        self.optical_axis_vector = None
        self.eyeball_rotation = None

        self._base_eye_sphere = self._generate_coordinates()

    def _generate_coordinates(self, resolution=13):
        """
        Generate the base coordinates for the eye sphere mesh.

        Args:
            resolution (int, optional): Sampling resolution of the sphere mesh. Default is 20.

        Returns:
            tuple: Meshgrid arrays (x, y, z) representing the sphere surface.
        """

        # Sample points along the surface of a unit sphere in spherical coordinates.
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi - np.pi / 8, resolution)

        # Convert spherical coordinates to Cartesian coordinates.
        x = self.eyeball_radius * np.outer(np.cos(u), np.sin(v))
        y = self.eyeball_radius * np.outer(np.sin(u), np.sin(v))
        z = self.eyeball_radius * np.outer(np.ones_like(u), np.cos(v))

        # Collect the coordinates into a single array.
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

        # The generated sphere opening is by default oriented downwards, along the y-axis.
        # So, we rotate it to have the front of the eye facing forward along the z-axis in
        # its initial state.
        rot_x = R.from_euler("x", 90, degrees=True)
        rotated = rot_x.apply(pts)
        x_r, y_r, z_r = rotated[:, 0], rotated[:, 1], rotated[:, 2]

        # Return the base shape of the eye sphere.
        return x_r.reshape(x.shape), y_r.reshape(y.shape), z_r.reshape(z.shape)

    def update(self, eyeball_center, optical_axis_vector):
        """
        Effectively updates the _transformed_eye_sphere attribute based on new eyeball center and optical axis estimates from Neon.
        """

        self.eyeball_center = eyeball_center
        self.optical_axis_vector = optical_axis_vector

        self._compute_eyeball_rotation()
        self._transform()

    def _compute_eyeball_rotation(self):
        """
        Compute the rotation required to align the eye sphere's forward direction with the optical axis.
        Sets the eyeball_rotation attribute for subsequent transformations.
        """

        axis_x, axis_y, axis_z = self.optical_axis_vector
        optical_axis_norm = np.linalg.norm(self.optical_axis_vector)

        # Transform the default Cartesian coordinates of the optical axis vector
        # to spherical coordinates.
        azimuth = np.arctan2(axis_y, axis_x)  # Rotation about Z axis
        elevation = np.arcsin(axis_z / optical_axis_norm)  # Rotation about Y axis

        azimuth -= np.pi / 2  # 0 azimuth is forward for Neon

        # Use scipy's rotation methods to create a rotation object that can be easily
        # applied later to the eye sphere coordinates.
        self.eyeball_rotation = R.from_euler("zx", [azimuth, elevation], degrees=False)

    def _transform(self):
        """
        Apply the computed rotation to the eye sphere and translate it to the current eye center.
        Updates the _transformed_eye_sphere attribute.
        """

        (x0, y0, z0) = self._base_eye_sphere

        # It is easiest to apply scipy rotation objects to numpy arrays,
        # so first, collect the coordinates into a single array.
        pts = np.stack([x0.ravel(), y0.ravel(), z0.ravel()], axis=1)

        # Apply the rotation to the base eye sphere coordinates.
        rotated = self.eyeball_rotation.apply(pts)
        x_r, y_r, z_r = rotated[:, 0], rotated[:, 1], rotated[:, 2]

        # The base eye sphere is by default at the origin, so produce
        # a translated version that sits at the current eye center estimate
        # from Neon.
        x_r += self.eyeball_center[0]
        y_r += self.eyeball_center[1]
        z_r += self.eyeball_center[2]

        # Save the transformed eye sphere for plotting later.
        self._transformed_eye_sphere = (
            x_r.reshape(x0.shape),
            y_r.reshape(y0.shape),
            z_r.reshape(z0.shape),
        )

    def plot(self, ax):
        """
        Plots the transformed eye sphere on the given 3D axis.
        """

        ax.plot_surface(
            self._transformed_eye_sphere[0],
            self._transformed_eye_sphere[1],
            self._transformed_eye_sphere[2],
            color="black",
            edgecolor="darkgray",
            shade=False,
            linewidths=0.8,
        )


class Pupil:
    """
    Represents the pupil of the eye as a disk.

    Attributes:
        eyeball_radius (float): The radius of the eyeball.
        eyeball_center (np.ndarray): The center point of the eyeball.
        pupil_diameter (float): The diameter of the pupil.
        eyeball_rotation (scipy.spatial.transform.Rotation): The rotation of the eyeball, as derived from the optical axis.
    """

    def __init__(self, eyeball_radius, eyeball_center, pupil_diameter):
        self.eyeball_radius = eyeball_radius
        self.eyeball_center = eyeball_center
        self.pupil_diameter = pupil_diameter

        # These will be computed when the `update()` method is called.
        self.eyeball_rotation = None
        self._transformed_pupil_disk = None

        self._base_pupil_disk = self._generate_coordinates()

    def _generate_coordinates(self, resolution=10):
        """
        Generate the coordinates for a flat disk representing the pupil.

        Args:
            resolution (int, optional): Number of points for the disk mesh. Default is 10.

        Returns:
            tuple: Meshgrid arrays (X, Y, Z) representing the pupil disk.
        """

        # A disk is just a flat, filled circle, so sample positions on
        # the surface of a disk in polar coordinates.
        r = np.linspace(0, 1.0, resolution)
        theta = np.linspace(0, 2 * np.pi, resolution)
        R, T = np.meshgrid(r, theta)

        # Convert to 3D Cartesian coordinates.
        X = R * np.cos(T)
        Y = np.zeros_like(X)  # Flat disk in XZ plane (y = 0)
        Z = R * np.sin(T)

        return X, Y, Z

    def update(self, eyeball_center, eyeball_rotation, pupil_diameter):
        """
        Effectively updates the _transformed_pupil_disk attribute based on new eyeball center, eyeball rotation, pupil diameter estimates from Neon.
        """

        self.eyeball_center = eyeball_center
        self.eyeball_rotation = eyeball_rotation
        self.pupil_diameter = pupil_diameter

        self._transfrom()

    def _transfrom(self):
        """
        Transform the pupil disk by scaling, rotating, and translating it to the correct position on the eye.
        Updates the _transformed_pupil_disk attribute.
        """

        (X, Y, Z) = self._base_pupil_disk

        # First, scale the pupil disk by pupil diameter.
        X_s = X * (self.pupil_diameter / 2)
        Y_s = Y * (self.pupil_diameter / 2)
        Z_s = Z * (self.pupil_diameter / 2)

        pts = np.stack([X_s.ravel(), Y_s.ravel(), Z_s.ravel()], axis=1)

        # The pupil sits on the surface of the eye sphere (local coordinates).
        pts += [0.0, self.eyeball_radius + 2.0, 0.0]

        # Rotate the pupil into position, according to the optical axis rotation.
        rot_matrix = (self.eyeball_rotation).as_matrix()
        pts = pts @ rot_matrix.T

        # Shift it to where the eye is centered in scene camera coordinates.
        pts += self.eyeball_center

        X_t = pts[:, 0].reshape(X.shape)
        Y_t = pts[:, 1].reshape(Y.shape)
        Z_t = pts[:, 2].reshape(Z.shape)

        self._transformed_pupil_disk = (X_t, Y_t, Z_t)

    def plot(self, ax):
        """
        Plots the 3D surface of the pupil on the given 3D axis.
        """

        ax.plot_surface(
            self._transformed_pupil_disk[0],
            self._transformed_pupil_disk[1],
            self._transformed_pupil_disk[2],
            color="black",
            shade=False,
        )


class EyeLid:
    """
    Represents an eyelid of the eye model.

    Attributes:
        eyeball_radius (float): The radius of the eyeball.
        eyeball_center (np.ndarray): The center position of the eyeball.
        eye_lid_angle (float): The angle of the eyelid, with respect to the scene camera's horizontal plane.
    """

    def __init__(self, eyeball_radius, eyeball_center, eye_lid_angle):
        self.eyeball_radius = eyeball_radius
        self.eyeball_center = eyeball_center
        self.eye_lid_angle = eye_lid_angle

        # This will be computed when the `update()` method is called.
        self._transformed_eye_lid = None

        self._base_eye_lid = self._generate_coordinates()

    def _generate_coordinates(self):
        """
        Generate the coordinates for an eyelid as a tube along a circular arc.

        Returns:
            tuple: Meshgrid arrays (X, Y, Z) representing the eyelid tube.
        """

        # Tube along circular arc
        theta = np.linspace(np.pi / 8, np.pi - np.pi / 8, 10)  # curve angle
        R = self.eyeball_radius + 1.5  # major radius (curve path)
        r = 0.45  # tube radius (thickness)
        n_circle = 5  # resolution of circular cross-section

        # Cross-section circle angles
        phi = np.linspace(0, 2 * np.pi, n_circle)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # Allocate arrays
        X = np.zeros((n_circle, len(theta)))
        Y = np.zeros_like(X)
        Z = np.zeros_like(X)

        # Build tube by sweeping the cross-section
        for i, t in enumerate(theta):
            # Center point on the arc
            cx = R * np.cos(t)
            cy = R * np.sin(t)
            cz = 0

            # Tangent vector (derivative of arc)
            tx = -R * np.sin(t)
            ty = R * np.cos(t)
            tz = 0
            tangent = np.array([tx, ty, tz])
            tangent /= np.linalg.norm(tangent)

            # Normal vector (pointing in Z)
            normal = np.array([0, 0, 1])

            # Binormal = tangent × normal
            binormal = np.cross(tangent, normal)

            # Build circular cross-section at this position
            for j in range(n_circle):
                X[j, i] = cx + r * (cos_phi[j] * normal[0] + sin_phi[j] * binormal[0])
                Y[j, i] = cy + r * (cos_phi[j] * normal[1] + sin_phi[j] * binormal[1])
                Z[j, i] = cz + r * (cos_phi[j] * normal[2] + sin_phi[j] * binormal[2])

        return X, Y, Z

    def update(self, eyeball_center, eye_lid_angle):
        """
        Effectively updates the _transformed_eye_lid attribute based on new eyeball center and eyelid angle estimates from Neon.
        """

        self.eyeball_center = eyeball_center
        self.eye_lid_angle = eye_lid_angle

        self._transform()

    def _transform(self):
        """
        Rotate and position the eyelid according to the specified angle and eye center.

        Args:
            eye_lid_angle (float): Angle by which to rotate the eyelid, with respect to the scene camera's horizontal plane.

        Returns:
            tuple: Transformed (X, Y, Z) arrays for the eyelid.
        """

        (x, y, z) = self._base_eye_lid

        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

        rot = R.from_euler("x", self.eye_lid_angle).as_matrix()
        rotated = pts @ rot.T

        X_rot = rotated[:, 0].reshape(x.shape)
        Y_rot = rotated[:, 1].reshape(y.shape)
        Z_rot = rotated[:, 2].reshape(z.shape)

        X_rot += self.eyeball_center[0]
        Y_rot += self.eyeball_center[1]
        Z_rot += self.eyeball_center[2]

        # Shift the eyelids forward a bit on the Matplotlib forward
        # axis. Otherwise, they blend a bit into the eyeball sphere.
        Y_rot += 9.0

        self._transformed_eye_lid = (X_rot, Y_rot, Z_rot)

    def plot(self, ax):
        """
        Plots the 3D surface of the eyelid on the given 3D axis.
        """

        ax.plot_surface(
            self._transformed_eye_lid[0],
            self._transformed_eye_lid[1],
            self._transformed_eye_lid[2],
            color="white",
            linewidth=2,
            rstride=1,
            cstride=1,
            shade=False,
        )


class ThreeDEyeModel:
    """
    A class for generating and manipulating a 3D model of an eye, including its center, orientation, pupil size, and eyelids.

    This class provides methods for updating these parameters, as well as for plotting the eye in 3D using matplotlib.

    Attributes:
        eyeball_radius (float): Radius of the eye [mm]; default is 12mm, the adult human average.
        eyeball_center (np.ndarray): Center position of the eye [mm].
        pupil_diameter (float): Diameter of the pupil [mm].
        eyelid_top_angle (float): Opening angle of the top eyelid [rad].
        eyelid_bottom_angle (float): Opening angle of the bottom eyelid [rad].
        optical_axis_vector (np.ndarray): Unit direction vector representing the optical axis.
    """

    def __init__(self, eyeball_radius=12):
        """
        Initialize the ThreeDEye object with default or specified radius.
        Generates the base geometry for the eye sphere, pupil, and eyelids.

        Args:
            eyeball_radius (float, optional): The (radius [mm]) of the eye sphere. Default is 12 mm, the adult human average.
        """

        self.eyeball_radius = eyeball_radius

        # Dummy default values for initializing & testing.
        self.eyeball_center = np.array([0.0, 0.0, 0.0])
        self.pupil_diameter = 4.0
        self.eyelid_top_angle = 0.2
        self.eyelid_bottom_angle = -0.2
        self.optical_axis_vector = np.array([0.0, 1.0, 0.0])

        # Instantiate the different classes that represent different parts of the eye model.
        self.optical_axis = OpticalAxis(self.eyeball_center, self.optical_axis_vector)
        self.eyeball = EyeBall(self.eyeball_center, self.eyeball_radius)
        self.pupil = Pupil(
            self.eyeball_radius, self.eyeball_center, self.pupil_diameter
        )
        self.top_eye_lid = EyeLid(
            self.eyeball_radius, self.eyeball_center, self.eyelid_top_angle
        )
        self.bottom_eye_lid = EyeLid(
            self.eyeball_radius, self.eyeball_center, self.eyelid_bottom_angle
        )

    def update(
        self,
        eyeball_center,
        optical_axis_vector,
        pupil_diameter,
        eyelid_top_angle,
        eyelid_bottom_angle,
    ):
        """
        Update the eye's parameters and recompute its geometry.

        Args:
            eyeball_center (np.ndarray): The new center position of the eye (3D vector).
            optical_axis_vector (np.ndarray): The new optical axis direction (3D vector).
            pupil_diameter (float): The new diameter of the pupil.
            eyelid_top_angle (float): The new angle for the top eyelid.
            eyelid_bottom_angle (float): The new angle for the bottom eyelid.
        """

        self.eyeball_center = eyeball_center

        # Match matplotlib conventions.
        self.eyeball_center[1] *= -1
        self.eyeball_center[2] *= -1

        # Match matplotlib conventions.
        self.optical_axis_vector = np.array(
            [
                optical_axis_vector[0],
                optical_axis_vector[2],
                -1 * optical_axis_vector[1],
            ]
        )

        self.pupil_diameter = pupil_diameter

        self.eyelid_top_angle = eyelid_top_angle
        self.eyelid_bottom_angle = eyelid_bottom_angle

        # Now that all the attributes are appropriately set, update each of the eye model components.
        self.eyeball.update(self.eyeball_center, self.optical_axis_vector)
        self.optical_axis.update(self.eyeball_center, self.optical_axis_vector)
        self.pupil.update(
            self.eyeball_center, self.eyeball.eyeball_rotation, self.pupil_diameter
        )
        self.top_eye_lid.update(self.eyeball_center, self.eyelid_top_angle)
        self.bottom_eye_lid.update(self.eyeball_center, self.eyelid_bottom_angle)

    def plot(self, ax, optical_axis_color):
        """
        Plot the full 3D eye model on the given matplotlib 3D axis.

        Args:
            ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis to plot on.
            optical_axis_color (str or tuple): Color for the optical axis arrow.
        """

        self.eyeball.plot(ax)
        self.optical_axis.plot(ax, optical_axis_color)
        self.pupil.plot(ax)
        self.top_eye_lid.plot(ax)
        self.bottom_eye_lid.plot(ax)
