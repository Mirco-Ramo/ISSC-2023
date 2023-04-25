import uuid

import numpy as np
from sqlalchemy import Column, Integer, String, ForeignKey, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

# UUID generation function
generate_uuid = lambda: str(uuid.uuid4().hex[:12])

country_iso_codes = {
    "china": "CN",
    "india": "IN",
    "france": "FR",
    "brazil": "BR",
    "mexico": "MX",
    "austria": "AT",
    "ireland": "IE",
    "england": "GB",
    "united states": "US",
    "united kingdom": "GB"
}


def euclid_distance(x_values, y_values):
    x_values = np.array(x_values)
    y_values = np.array(y_values)

    ed = np.linalg.norm(np.array(x_values) - np.array(y_values))
    return ed


def update_finger_table(session):
    fingers = [(3, "ring", "The ring finger"),
               (4, "pinky", "The pinky finger"),
               (0, "thumb", "The thumb finger"),
               (1, "index", "The index finger"),
               (2, "middle", "The middle finger"), ]

    for f_id, finger, desc in fingers:
        new_finger = Finger(finger_id=f_id,
                            finger=finger,
                            description=desc,
                            uuid=generate_uuid(), )
        session.add(new_finger)
    session.commit()


def update_experiment_table(session):
    experiments = [(3, "Single numeric characters on iOS devices"),
                   (4, "Multiple alphanumeric characters on iOS devices"),
                   (1, "Single alphanumeric characters on Android devices"),
                   (2, "Multiple alphanumeric characters on Android devices"), ]

    for exp_id, desc in experiments:
        new_experiment = Experiment(experiment_id=exp_id,
                                    description=desc,
                                    uuid=generate_uuid(), )
        session.add(new_experiment)
    session.commit()


# Model Definitions
# -----------------
class Finger(Base):
    __tablename__ = "finger"

    uuid = Column(String, unique=True, name="uuid", index=True, primary_key=True, )
    finger = Column(String)
    finger_id = Column(Integer)
    description = Column(String)


class Experiment(Base):
    __tablename__ = "experiment"

    uuid = Column(String, unique=True, name="uuid", index=True, primary_key=True, )
    description = Column(String)
    experiment_id = Column(Integer)


class Touch(Base):
    __tablename__ = "touch"
    x = Column(Float)
    y = Column(Float)
    timestamp = Column(Integer)
    stroke_id = Column(String, ForeignKey('stroke.uuid'), index=True)
    uuid = Column(String, unique=True, name="uuid", index=True, primary_key=True, )


class Stroke(Base):
    __tablename__ = "stroke"
    touches = relationship("Touch", backref="stroke")
    glyph_id = Column(String, ForeignKey('glyph.uuid'), index=True)
    uuid = Column(String, unique=True, name="uuid", index=True, primary_key=True, )

    @property
    def duration(self):
        """Duration in femtoseconds"""
        touch_start_time = self.touches[0].timestamp
        touch_end_time = self.touches[-1].timestamp
        return touch_end_time - touch_start_time

    @property
    def x_values(self):
        return [t.x for t in self.touches]

    @property
    def y_values(self):
        return [t.y for t in self.touches]

    @property
    def x_y_values(self):
        return [(t.x, t.y) for t in self.touches]

    def arc_length(self, mode="euclid"):
        if mode == "euclid":
            arc_length = euclid_distance(self.x_values, self.y_values)
        else:
            pass  # Tbd (Bezier)

        return arc_length


class Glyph(Base):
    __tablename__ = "glyph"

    ground_truth = Column(String)
    stroke_delays = Column(String)

    strokes = relationship("Stroke", backref="glyph")
    uuid = Column(String, unique=True, name="uuid", index=True, primary_key=True, )
    glyph_sequence_id = Column(String, ForeignKey('glyph_sequence.uuid'), index=True)

    @property
    def num_of_touches(self):
        return sum([len(stroke.touches) for stroke in self.strokes])

    @property
    def duration(self):
        """Duration in seconds"""
        stroke_delays = sum((abs(float(i)) for i in self.stroke_delays.split(" ")))
        strokes_duration = sum((i.duration for i in self.strokes))
        return round(((stroke_delays + strokes_duration) * 1e-15), 6)

    @property
    def coordinates(self):
        """The glyph's coordinates in 2D space"""
        x = [stroke.x_values for stroke in self.strokes]
        y = [stroke.y_values for stroke in self.strokes]
        return x, y

    @property
    def x_y_values(self):
        return [stroke.x_y_values for stroke in self.strokes]

    def serialize(self, session):
        """JSON-serialize glyph"""

        ng = {'strokes': [], 'uuid': self.uuid, 'ground_truth': self.ground_truth,
              'stroke_delays': self.stroke_delays}  # new glyph

        ngs = session.query(GlyphSequence).filter(
            GlyphSequence.uuid == self.glyph_sequence_id).first()  # Glyph sequence

        ng['ar'] = ngs.aspect_ratio
        ng['subject_id'] = ngs.subject_id
        ng['client_height'] = ngs.client_height

        for stroke in self.strokes:
            ns = {'touches': [], 'uuid': stroke.uuid}  # new stroke

            for touch in stroke.touches:
                nt = {'x': touch.x, 'y': touch.y, 'uuid': touch.uuid, 'timestamp': touch.timestamp}  # new touch

                ns['touches'].append(nt)
            ng['strokes'].append(ns)

        return ng

    def __repr__(self):
        return f"Glyph ({self.ground_truth}) ({self.uuid})"


class GlyphSequence(Base):
    __tablename__ = "glyph_sequence"

    device = Column(String)
    finger = Column(Integer)
    experiment = Column(Integer)
    aspect_ratio = Column(Float)
    client_height = Column(Float)
    ground_truth = Column(String)
    glyph_delays = Column(String)
    glyph_indices = Column(String)

    glyphs = relationship("Glyph", backref="glyph_sequence")
    subject_id = Column(String, ForeignKey('subject.uuid'), index=True)
    uuid = Column(String, unique=True, name="uuid", index=True, primary_key=True, )

    @property
    def ground_truth_length(self):
        return len(self.ground_truth)

    def __repr__(self):
        return f"Glyph Sequence ({self.ground_truth}) ({self.uuid})"


class Subject(Base):
    __tablename__ = "subject"

    uuid = Column(String, unique=True, name="uuid", index=True, primary_key=True, )
    old_id = Column(Integer)
    age = Column(Integer)
    sex = Column(Boolean)
    handedness = Column(Boolean)
    nationality = Column(String)
    glyph_sequences = relationship("GlyphSequence", backref="subject")

    def __repr__(self):
        return f"Subject ({self.uuid})"
