# court.py
from typing import NamedTuple, List, Tuple
from point import Point3D

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


class Court:
    def __init__(self, rule_type="FIBA"):
        self.court_definition = court_definitions[rule_type]

    def get_world_points(self) -> List[Tuple[str, Point3D]]:
        w, h = self.court_definition.width, self.court_definition.height
        three_point_limit_top = self.court_definition.three_point_limit
        three_point_limit_bottom = h - self.court_definition.three_point_limit
        return [
            ("Bottom Left Corner", Point3D(0, 0, 0)),
            ("Bottom Right Corner", Point3D(w, 0, 0)),
            ("Top Right Corner", Point3D(w, h, 0)),
            ("Top Left Corner", Point3D(0, h, 0)),
            (
                "Left Free Throw Line (Bottom)",
                Point3D(
                    self.court_definition.key_area_length,
                    h / 2 - self.court_definition.key_area_width / 2,
                    0,
                ),
            ),
            (
                "Right Free Throw Line (Bottom)",
                Point3D(
                    w - self.court_definition.key_area_length,
                    h / 2 - self.court_definition.key_area_width / 2,
                    0,
                ),
            ),
            (
                "Left Free Throw Line (Top)",
                Point3D(
                    self.court_definition.key_area_length,
                    h / 2 + self.court_definition.key_area_width / 2,
                    0,
                ),
            ),
            (
                "Right Free Throw Line (Top)",
                Point3D(
                    w - self.court_definition.key_area_length,
                    h / 2 + self.court_definition.key_area_width / 2,
                    0,
                ),
            ),
            ("Left 3-Point Corner", Point3D(0, three_point_limit_top, 0)),
            ("Right 3-Point Corner", Point3D(w, three_point_limit_top, 0)),
            ("Top Left 3-Point Corner", Point3D(0, three_point_limit_bottom, 0)),
            ("Top Right 3-Point Corner", Point3D(w, three_point_limit_bottom, 0)),
            ("Center of Court", Point3D(w / 2, h / 2, 0)),
            (
                "Top of Left 3-Point Arc",
                Point3D(
                    self.court_definition.rim_center_offset,
                    h / 2 + self.court_definition.three_point_distance,
                    0,
                ),
            ),
            (
                "Top of Right 3-Point Arc",
                Point3D(
                    w - self.court_definition.rim_center_offset,
                    h / 2 + self.court_definition.three_point_distance,
                    0,
                ),
            ),
            ("Center Line Left Intersection", Point3D(0, h / 2, 0)),
            ("Center Line Right Intersection", Point3D(w, h / 2, 0)),
        ]
