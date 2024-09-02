from typing import NamedTuple

import cv2
import numpy as np

# from calib3d import ProjectiveDrawer, Point2D, Point3D, Calib

court_dim = {
    "NBA": (2865.0, 1524.0),
    "FIBA": (2800.0, 1500.0),
    "NFHS30": (2560.32, 1524.0),
    "IH_IIHF": (6096.0, 2590.0),
    "NCAA15M": (2865.12, 1524.0),
    "NCAA15W": (2865.12, 1524.0),
    "NCAAM": (2865.12, 1524.0),
    "NCAAW": (2865.12, 1524.0),
}

court_types_from_rule_type = {
    "NFHS30": "NFHS",
    "NCAA15M": "NCAA",
    "NCAA15W": "NCAA",
    "NCAAM": "NCAA",
    "NCAAW": "NCAA",
}


BALL_DIAMETER = 23.8


class CourtDefinition(NamedTuple):
    width: float
    height: float
    circle_diameter: float
    three_point_distance: float
    three_point_limit: float
    key_area_width: float
    key_area_length: float
    board_offset: float
    board_width: float
    board_height: float
    board_elevation: float
    rim_center_offset: float
    rim_height: float
    rim_radius: float
    no_charge_zone_radius: float


#                                                       KEY AREA
#                                                        width
#                                             <--------------------------->
#    +----------+----------------------------+-----------------------------+----------------------------+------------+
#    |          |                        ^   |     ^ BOARD            ^    |                            |            |
#    |          |                        |   |     | offset           | RIM                             |            |
#    |          |                        |   |     v   ------------   | offset                          |            |
#    |          |                        |   |            ( X )       v    |                            |            |
#    | 3-POINTS |                        |   |             ˘-\             |                            |            |
#    |  limit   |               KEY AREA |   |                \            |                            |            |
#    |<-------->|                lenght  |   |                 \           |                            |            |
#    |          |                        |   |                  \          |                            |            |
#    |           |                       |   |                   \         |                           |             |
#    |           |                       |   |                    \        |                           |             |
#    |            \                      |   |                     \       |                          /              |
#    |             |                     |   |                      \      |                         |               |
#    |              \                    v   |                       \     |                        /                |
#    |               \                       +-----+-----------------+\----+                       /                 |
#    |                 \                           |<--------------->| \                         /                   |
#    |                   '.                         \    CIRCLE d   /   \ 3-POINTS            .'                     |
#    |                     '-.                        '.         .'      \  dist           .-'                       |
#    |                        '-.                        `-----´          \             .-'                          |
#    |                           '--.                                      \        .--'                             |
#    |                               '--.                                   \   .--'                                 |
#    |                                   '---___                         ___-ˇ-'                                     |
#    |                                          ''-------_______-------''                                            |
#    |                                                                                                               |
# TODO: Warning /!\ Those numbers should be checked carefully if model rely on them
#                           |      COURT      | CIRCLE |    3-POINTS  |    KEY AREA    |          BOARD                      |        RIM        | NO_CHARGE
court_definitions = {  # | width  | height |   d    | dist , limit | width | length | offset | width | height | elevation | offset |  h  | r  | radius
    "FIBA": CourtDefinition(
        2800.0,
        1500.0,
        360.0,
        675,
        90.0,
        490.0,
        575.0,
        122,
        183,
        106.6,
        290,
        160,
        305,
        23,
        125,
    ),
    "NBA": CourtDefinition(
        2865.1,
        1524.0,
        366.0,
        723.9,
        90.0,
        488.0,
        575.0,
        122,
        183,
        106.6,
        290,
        160,
        305,
        23,
        122,
    ),
    "NCAA": CourtDefinition(
        2865.1,
        1524.0,
        366.0,
        723.9,
        90.0,
        488.0,
        575.0,
        122,
        183,
        106.6,
        290,
        157.5,
        305,
        23,
        122,
    ),
    "NCAAW": CourtDefinition(
        2865.1,
        1524.0,
        366.0,
        632,
        90.0,
        488.0,
        575.0,
        122,
        183,
        106.6,
        290,
        157.5,
        305,
        23,
        122,
    ),
    "NFHS": CourtDefinition(
        2865.1,
        1524.0,
        366.0,
        602,
        160.0,
        488.0,
        574.0,
        122,
        183,
        106.6,
        290,
        157.5,
        305,
        23,
        122,
    ),
}
# https://en.wikipedia.org/wiki/Basketball_court#Table
# https://en.wikipedia.org/wiki/Key_(basketball)


# %%


class ProjectiveDrawer:
    """Given the calibration information with `Calib`, and a number of segments
    to decompose straight lines, this objet offer several functions to draw
    on a 2D canvas given 3D coordinates.
    """

    def __init__(self, calib: Calib, color, thickness: int = 1, segments: int = 10):
        self.color = color
        self.thickness = thickness
        self.calib = calib
        self.segments = segments

    def _polylines(
        self,
        canvas,
        points: Point2D,
        color=None,
        thickness: int = None,
        markersize=None,
        **kwargs
    ):
        thickness = thickness or self.thickness
        color = color or self.color
        if isinstance(canvas, np.ndarray):
            points = np.array(points.astype(np.int32).T.reshape((-1, 1, 2)))
            if thickness < 0:
                cv2.fillPoly(canvas, [points], color=color, **kwargs)
            else:
                cv2.polylines(
                    canvas, [points], False, color=color, thickness=thickness, **kwargs
                )
            if markersize:
                for point in points:
                    cv2.drawMarker(
                        canvas,
                        tuple(point[0]),
                        color=color,
                        markerSize=markersize,
                        thickness=thickness,
                    )
        else:
            if thickness < 0:
                try:
                    import matplotlib as mpl

                    p = mpl.patches.Polygon(
                        np.array(list(zip(points.x, points.y))),
                        facecolor=np.array(color) / 255,
                        alpha=0.35,
                    )
                    canvas.add_patch(p)
                except ImportError:
                    raise ImportError("The current implementation requires matplotlib")
            else:
                canvas.plot(
                    points.x,
                    points.y,
                    linewidth=thickness,
                    color=np.array(color) / 255,
                    markersize=markersize,
                    **kwargs
                )

    def polylines(
        self, canvas, points3D: Point3D, color=None, thickness: int = None, **kwargs
    ):
        self._polylines(
            canvas,
            self.calib.project_3D_to_2D(points3D),
            color=color,
            thickness=thickness,
            **kwargs
        )
        # for point3D1, point3D2 in zip(points3D, points3D.close()[:,1:]):
        #    self.draw_line(canvas, point3D1, point3D2, *args, **kwargs)

    def draw_line(
        self,
        canvas,
        point3D1: Point3D,
        point3D2: Point3D,
        color=None,
        thickness: int = None,
        only_visible=True,
        **kwargs
    ):
        if only_visible:
            try:
                point3D1, point3D2 = visible_segment(self.calib, point3D1, point3D2)
            except ValueError:
                return
        points3D = Point3D(np.linspace(point3D1, point3D2, self.segments + 1))
        self._polylines(
            canvas,
            self.calib.project_3D_to_2D(points3D),
            color=color,
            thickness=thickness,
            **kwargs
        )

    def draw_arc(
        self,
        canvas,
        center,
        radius,
        start_angle=0.0,
        stop_angle=2 * np.pi,
        color=None,
        thickness=None,
    ):
        thickness = thickness or self.thickness
        angles = np.linspace(start_angle, stop_angle, self.segments * 4 + 1)
        xs = np.cos(angles) * radius + center.x
        ys = np.sin(angles) * radius + center.y
        zs = np.ones_like(angles) * center.z
        points3D = Point3D(np.vstack((xs, ys, zs)))
        self._polylines(
            canvas,
            self.calib.project_3D_to_2D(points3D),
            color=color,
            thickness=thickness,
        )

    def draw_rectangle(self, canvas, point3D1, point3D2):
        c1 = point3D1
        c3 = point3D2
        if point3D1.z == point3D2.z:
            c2 = Point3D(c1.x, c3.y, c1.z)
            c4 = Point3D(c3.x, c1.y, c1.z)
        elif point3D1.x == point3D2.x:
            c2 = Point3D(c1.x, c1.y, c3.z)
            c4 = Point3D(c1.x, c3.y, c1.z)
        elif point3D1.y == point3D2.y:
            c2 = Point3D(c1.x, c1.y, c3.z)
            c4 = Point3D(c3.x, c1.y, c1.z)
        corners = [c1, c2, c3, c4, c1]
        for p1, p2 in zip(corners, corners[1:]):
            self.draw_line(canvas, p1, p2)

    def fill_polygon(self, canvas, points3D):
        points3D = points3D.close().linspace(self.segments)
        self._polylines(canvas, self.calib.project_3D_to_2D(points3D), thickness=-1)


# %%
class Court:
    """Defines the world 3D coordinates of a Basketball court.
    Please refer to calib.py if you want to obtain their 2D image coordinates
    """

    def __init__(self, rule_type="FIBA"):
        self.w, self.h = court_dim[rule_type]
        court_type = court_types_from_rule_type.get(rule_type, rule_type)
        self.court_definition = court_definitions[court_type]

    @property
    def corners(self):
        return Point3D([0, self.w, self.w, 0], [0, 0, self.h, self.h], [0, 0, 0, 0])

    @property
    def edges(self):
        for c1, c2 in zip(self.corners, self.corners.close()[:, 1:]):
            yield c1, c2

    def visible_edges(self, calib: Calib):
        for edge in self.edges:
            try:
                yield calib.visible_edge(edge)
            except ValueError:
                continue

    def projects_in(self, calib: Calib, point2D):
        point3D = calib.project_2D_to_3D(point2D, Z=0)
        return (
            point3D.x >= 0
            and point3D.x <= self.w
            and point3D.y >= 0
            and point3D.y <= self.h
        )

    def draw_rim(self, image, calib: Calib, color=(128, 200, 90), thickness=2):
        pd = ProjectiveDrawer(calib, color, thickness=thickness)
        offset = self.court_definition.rim_center_offset
        h = self.court_definition.rim_height
        r = self.court_definition.rim_radius
        pd.draw_arc(image, Point3D(offset, self.h / 2, -h), r)
        pd.draw_arc(image, Point3D(self.w - offset, self.h / 2, -h), r)

    def draw_poles(self, image, calib: Calib, color=(10, 10, 10)):
        pass

    def draw_net(self, image, calib: Calib, color=(128, 90, 200)):
        offset = self.court_definition.rim_center_offset
        r = self.court_definition.rim_radius
        angles = np.linspace(0, np.pi * 2, 20)
        pd = ProjectiveDrawer(calib, color)
        for x in [offset, self.w - offset]:  # two rims
            center = Point3D(x, self.h / 2, -self.court_definition.rim_height)
            xts = np.cos(angles) * r + center.x
            yts = np.sin(angles) * r + center.y
            xbs = np.cos(angles) * r / 2 + center.x
            ybs = np.sin(angles) * r / 2 + center.y

            zs = np.ones_like(angles) * center.z
            points = Point3D(
                [
                    Point3D(x, y, z)
                    for x, y, z in zip(
                        np.concatenate((xts, xbs)),
                        np.concatenate((yts, ybs)),
                        np.concatenate((zs, zs + 40)),
                    )
                ]
            )
            pd.polylines(image, points)

    def _get_three_points_anchors(self):
        def sign(value):
            return -1 if value < 0 else 1

        #    +----------+----------------------------+-----------------------------+----------------------------+------------+
        #    |          |                            |              ^              |                            |            |
        #    |          |                            |         RIM  |              |                            |            |
        #    |          |                            |       offset |              |                            |            |
        #    |          |                            |              v              |                            |            |
        #    |          |                            |       -----( X )            |                            |            |
        #    | 3-POINTS |                            | -------      ^              |                            |            |
        #    |  limit   |                      ------|-   |         |              |                            |            |
        #    |<-------->|               -------      |     \        |              |                            |            |
        #    |          |        -------   Z         |       '.     |              |                            |            |
        #    |           |-------                    |   alpha  `---|              |                           |             |
        #    |           |                           |              |              |                           |             |
        #    |            \                          |              | 3-POINTS     |                          /              |
        #    |             |                         |              |   dist       |                         |               |
        #    |              \                        |              |              |                        /                |
        #    |               \                       +-----+-----------------+-----+                       /                 |
        #    |                 \                           |        |        |                           /                   |
        #    |                   '.                         \       |       /                         .'                     |
        #    |                     '-.                        '.    |    .'                        .-'                       |
        #    |                        '-.                        `-----´                        .-'                          |
        #    |                           '--.                       |                       .--'                             |
        #    |                               '--.                   |                   .--'                                 |
        #    |                                   '---___            |            ___-ˇ-'                                     |
        #    |                                          ''-------___v____-------''                                           |
        #    |                                                                                                               |

        # The goal of this function is to determine the "alpha" angle (see illustration above).
        # alpha: angle (in radian) between a a line at the center of the court height, and where the 3-points arc meets the straight line (nearby the "3-POINTS limit" annotation here above)
        # Y: line from the center of the rim to the point where the 3-points arc meets the straight line (nearby the "3-POINTS limit" annotation here above)

        # See https://mathworld.wolfram.com/Circle-LineIntersection.html for terminology
        # Note that all the computation are done on a euclidean standard orthonormal basis (positive abscissa pointing to the right, positive ordinate pointing to the top)
        y1 = 0
        y2 = -self.w / 2  # Arbitary point on the extension of the 3-points line
        x1 = -(
            self.h / 2 - self.court_definition.three_point_limit
        )  # Considers the center of the 3-points circle at (0, 0), with y-axis inverted
        x2 = x1
        dx = x2 - x1
        dy = y2 - y1
        dr = np.sqrt(dx**2 + dy**2)
        det = x1 * y2 - x2 * y1
        discriminant = (self.court_definition.three_point_distance**2) * (
            dr**2
        ) - det**2
        assert (
            discriminant >= 0
        )  # Two points of intersection (one in the court, the other before the baseline) or 1 point intersection
        x_intersection = (det * dy + sign(dy) * dx * np.sqrt(discriminant)) / (dr**2)
        y_intersection = (-det * dx + abs(dy) * np.sqrt(discriminant)) / (dr**2)
        if (
            y_intersection >= 0
        ):  # The estimated intersection is before the rim, compute the second intersection
            x_intersection = (det * dy - sign(dy) * dx * np.sqrt(discriminant)) / (
                dr**2
            )
            y_intersection = (-det * dx - abs(dy) * np.sqrt(discriminant)) / (dr**2)

        # Compute alpha, the angle between the Z line and the vertical "3-points dist" line
        Z_line_vector = [0 - 0, 0 + self.court_definition.three_point_distance]
        three_point_dist_line_vector = [0 - x_intersection, 0 - y_intersection]
        alpha = np.arccos(
            np.dot(Z_line_vector, three_point_dist_line_vector)
            / (
                np.linalg.norm(Z_line_vector)
                * np.linalg.norm(three_point_dist_line_vector)
            )
        )

        return alpha, -y_intersection + self.court_definition.rim_center_offset

    def draw_lines(self, image, calib: Calib, color=(255, 255, 0), thickness=5):
        """
        If color is None, use incremental color
        """
        pd = ProjectiveDrawer(calib, color, thickness)

        N = 100  # maximum number of lines
        _color = iter(range(1, N)) if color is None else iter([color] * N)

        # draw court borders
        for edge in self.edges:  # self.visible_edges(calib):
            pd.draw_line(image, edge[0], edge[1], color=next(_color))

        # draw midcourt-line
        pd.draw_line(
            image,
            Point3D(self.w / 2, 0, 0),
            Point3D(self.w / 2, self.h, 0),
            color=next(_color),
        )

        # draw central circle
        r = self.court_definition.circle_diameter / 2
        pd.draw_arc(image, Point3D(self.w / 2, self.h / 2, 0), r, color=next(_color))

        # draw paints
        w, l = (
            self.court_definition.key_area_width,
            self.court_definition.key_area_length,
        )
        # left paint
        pd.draw_line(
            image,
            Point3D(0, self.h / 2 - w / 2, 0),
            Point3D(l, self.h / 2 - w / 2, 0),
            color=next(_color),
        )
        pd.draw_line(
            image,
            Point3D(l, self.h / 2 - w / 2, 0),
            Point3D(l, self.h / 2 + w / 2, 0),
            color=next(_color),
        )
        pd.draw_line(
            image,
            Point3D(0, self.h / 2 + w / 2, 0),
            Point3D(l, self.h / 2 + w / 2, 0),
            color=next(_color),
        )
        pd.draw_arc(
            image,
            Point3D(l, self.h / 2, 0),
            r,
            -np.pi / 2,
            np.pi / 2,
            color=next(_color),
        )
        # right paint
        pd.draw_line(
            image,
            Point3D(self.w, self.h / 2 - w / 2, 0),
            Point3D(self.w - l, self.h / 2 - w / 2, 0),
            color=next(_color),
        )
        pd.draw_line(
            image,
            Point3D(self.w - l, self.h / 2 - w / 2, 0),
            Point3D(self.w - l, self.h / 2 + w / 2, 0),
            color=next(_color),
        )
        pd.draw_line(
            image,
            Point3D(self.w, self.h / 2 + w / 2, 0),
            Point3D(self.w - l, self.h / 2 + w / 2, 0),
            color=next(_color),
        )
        pd.draw_arc(
            image,
            Point3D(self.w - l, self.h / 2, 0),
            r,
            np.pi / 2,
            3 * np.pi / 2,
            color=next(_color),
        )

        # draw 3-points line
        offset = self.court_definition.rim_center_offset
        r = self.court_definition.three_point_distance
        arc_rad, x_intersection = self._get_three_points_anchors()
        # left 3-points-line
        pd.draw_arc(
            image,
            Point3D(offset, self.h / 2, 0),
            r,
            -arc_rad,
            arc_rad,
            color=next(_color),
        )
        pd.draw_line(
            image,
            Point3D(0, self.court_definition.three_point_limit, 0),
            Point3D(x_intersection, self.court_definition.three_point_limit, 0),
            color=next(_color),
        )
        pd.draw_line(
            image,
            Point3D(0, self.h - self.court_definition.three_point_limit, 0),
            Point3D(
                x_intersection, self.h - self.court_definition.three_point_limit, 0
            ),
            color=next(_color),
        )
        # right 3-points-line
        pd.draw_arc(
            image,
            Point3D(self.w - offset, self.h / 2, 0),
            r,
            np.pi - arc_rad,
            np.pi + arc_rad,
            color=next(_color),
        )
        pd.draw_line(
            image,
            Point3D(self.w, self.court_definition.three_point_limit, 0),
            Point3D(
                self.w - x_intersection, self.court_definition.three_point_limit, 0
            ),
            color=next(_color),
        )
        pd.draw_line(
            image,
            Point3D(self.w, self.h - self.court_definition.three_point_limit, 0),
            Point3D(
                self.w - x_intersection,
                self.h - self.court_definition.three_point_limit,
                0,
            ),
            color=next(_color),
        )

    def fill_court(self, image, calib, color=(255, 0, 255)):
        pd = ProjectiveDrawer(calib, color)
        pd.fill_polygon(image, self.corners)

    def fill_court_coordinates(self, image, calib):
        pass

    @property
    def left_key_area(self):
        """Return the 3D coordinates of the 4 corners of the left key area"""
        return Point3D(
            [
                0,
                self.court_definition.key_area_length,
                self.court_definition.key_area_length,
                0,
            ],  # x
            [
                self.h / 2 - self.court_definition.key_area_width / 2,
                self.h / 2 - self.court_definition.key_area_width / 2,
                self.h / 2 + self.court_definition.key_area_width / 2,
                self.h / 2 + self.court_definition.key_area_width / 2,
            ],  # y
            [0, 0, 0, 0],
        )  # z

    @property
    def right_key_area(self):
        """Return the 3D coordinates of the 4 corners of the right key area"""
        return Point3D(
            [
                self.w - self.court_definition.key_area_length,
                self.w,
                self.w,
                self.w - self.court_definition.key_area_length,
            ],  # x
            [
                self.h / 2 - self.court_definition.key_area_width / 2,
                self.h / 2 - self.court_definition.key_area_width / 2,
                self.h / 2 + self.court_definition.key_area_width / 2,
                self.h / 2 + self.court_definition.key_area_width / 2,
            ],  # y
            [0, 0, 0, 0],
        )  # z

    @property
    def left_board(self):
        """Return the 3D coordinates of the 4 corners of the left basketball board (panel)"""
        board_offset = self.court_definition.board_offset
        board_width = self.court_definition.board_width
        board_height = self.court_definition.board_height
        board_elevation = self.court_definition.board_elevation
        return Point3D(
            [
                0 + board_offset,
                0 + board_offset,
                0 + board_offset,
                0 + board_offset,
            ],  # x
            [
                self.h / 2 - board_width / 2,
                self.h / 2 + board_width / 2,
                self.h / 2 + board_width / 2,
                self.h / 2 - board_width / 2,
            ],  # y
            [
                -board_elevation,
                -board_elevation,
                -board_elevation - board_height,
                -board_elevation - board_height,
            ],
        )  # z

    @property
    def right_board(self):
        """Return the 3D coordinates of the 4 corners of the right basketball board (panel)"""
        board_offset = self.court_definition.board_offset
        board_width = self.court_definition.board_width
        board_height = self.court_definition.board_height
        board_elevation = self.court_definition.board_elevation
        return Point3D(
            [
                self.w - board_offset,
                self.w - board_offset,
                self.w - board_offset,
                self.w - board_offset,
            ],  # x
            [
                self.h / 2 - board_width / 2,
                self.h / 2 + board_width / 2,
                self.h / 2 + board_width / 2,
                self.h / 2 - board_width / 2,
            ],  # y
            [
                -board_elevation,
                -board_elevation,
                -board_elevation - board_height,
                -board_elevation - board_height,
            ],
        )  # z

    def fill_board(self, image, calib: Calib, color=(128, 255, 128)):
        pd = ProjectiveDrawer(calib, color)
        pd.fill_polygon(image, self.left_board)
        pd.fill_polygon(image, self.right_board)


def sort_points_clockwise(data):
    assert data.size != 0, "Empty input data"
    mean = np.mean(data, axis=0)  # center of mass of the points
    angles = np.arctan2((data - mean)[:, 1], (data - mean)[:, 0])
    angles[angles < 0] = (
        angles[angles < 0] + 2 * np.pi
    )  # Transform angles from [-pi,pi] -> [0, 2*pi]
    return data[np.argsort(angles)]


def draw_ball(
    image,
    calib: Calib,
    center: Point3D,
    color=(0, 0, 255),
    cross=True,
    label=None,
    thickness=1,
    fontsize=10,
):
    pd = ProjectiveDrawer(calib, color, thickness=thickness)
    ground3D = Point3D(center.x, center.y, 0)
    pd.draw_line(image, center, ground3D, lineType=cv2.LINE_AA, color=color)
    if cross:
        pd.draw_line(
            image,
            ground3D + Point3D(100, 0, 0),
            ground3D - Point3D(100, 0, 0),
            lineType=cv2.LINE_AA,
            thickness=1,
            color=color,
        )
        pd.draw_line(
            image,
            ground3D + Point3D(0, 100, 0),
            ground3D - Point3D(0, 100, 0),
            lineType=cv2.LINE_AA,
            thickness=1,
            color=color,
        )
    radius = pd.calib.compute_length2D(center, BALL_DIAMETER / 2)[0]
    x, y = pd.calib.project_3D_to_2D(center).to_int_tuple()

    cv2.circle(image, (x, y), int(radius), color, 1)
    if label is not None:
        cv2.putText(
            image,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontsize,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
