"""Persona schema for event generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Persona:
    """A persona definition for generating realistic event streams.

    Personas define a person's characteristics, routines, and lifestyle
    to guide the LLM in generating coherent, realistic event sequences.

    Attributes:
        name: The persona's name.
        age: The persona's age.
        occupation: Their job or primary activity.
        location: Where they live (city/region).
        living_situation: e.g., "lives alone", "with family", "with roommates".
        work_schedule: Description of typical work hours.
        habits: List of regular habits and routines.
        interests: List of hobbies and interests.
        health_goals: Health and fitness related goals.
        social_circle: Description of social relationships.
        personality_traits: Key personality characteristics.
        typical_day: Brief description of a typical day.
        constraints: Any constraints or preferences (e.g., dietary, mobility).
    """

    name: str
    age: int
    occupation: str
    location: str = "San Francisco, CA"
    living_situation: str = "lives alone in apartment"
    work_schedule: str = "9am-5pm weekdays, remote work"
    habits: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)
    health_goals: list[str] = field(default_factory=list)
    social_circle: str = "small group of close friends, coworkers"
    personality_traits: list[str] = field(default_factory=list)
    typical_day: str = ""
    constraints: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Format the persona as a prompt string for the LLM.

        Returns:
            A formatted string describing the persona.
        """
        lines = [
            f"# Persona: {self.name}",
            "",
            "## Demographics",
            f"- Age: {self.age}",
            f"- Occupation: {self.occupation}",
            f"- Location: {self.location}",
            f"- Living situation: {self.living_situation}",
            "",
            "## Schedule & Routine",
            f"- Work schedule: {self.work_schedule}",
        ]

        if self.habits:
            lines.append("- Regular habits:")
            for habit in self.habits:
                lines.append(f"  - {habit}")

        if self.typical_day:
            lines.append(f"- Typical day: {self.typical_day}")

        lines.append("")
        lines.append("## Interests & Goals")

        if self.interests:
            lines.append("- Interests/hobbies:")
            for interest in self.interests:
                lines.append(f"  - {interest}")

        if self.health_goals:
            lines.append("- Health goals:")
            for goal in self.health_goals:
                lines.append(f"  - {goal}")

        lines.append("")
        lines.append("## Social & Personality")
        lines.append(f"- Social circle: {self.social_circle}")

        if self.personality_traits:
            lines.append(f"- Personality: {', '.join(self.personality_traits)}")

        if self.constraints:
            lines.append("")
            lines.append("## Constraints")
            for constraint in self.constraints:
                lines.append(f"- {constraint}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert persona to dictionary.

        Returns:
            Dictionary representation of the persona.
        """
        return {
            "name": self.name,
            "age": self.age,
            "occupation": self.occupation,
            "location": self.location,
            "living_situation": self.living_situation,
            "work_schedule": self.work_schedule,
            "habits": self.habits,
            "interests": self.interests,
            "health_goals": self.health_goals,
            "social_circle": self.social_circle,
            "personality_traits": self.personality_traits,
            "typical_day": self.typical_day,
            "constraints": self.constraints,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Persona:
        """Create a Persona from a dictionary.

        Args:
            data: Dictionary with persona fields.

        Returns:
            A Persona instance.
        """
        return cls(
            name=data["name"],
            age=data["age"],
            occupation=data["occupation"],
            location=data.get("location", "San Francisco, CA"),
            living_situation=data.get("living_situation", "lives alone in apartment"),
            work_schedule=data.get("work_schedule", "9am-5pm weekdays"),
            habits=data.get("habits", []),
            interests=data.get("interests", []),
            health_goals=data.get("health_goals", []),
            social_circle=data.get("social_circle", ""),
            personality_traits=data.get("personality_traits", []),
            typical_day=data.get("typical_day", ""),
            constraints=data.get("constraints", []),
        )

    def save(self, path: str | Path) -> None:
        """Save persona to a JSON file.

        Args:
            path: Path to save the persona.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Persona:
        """Load a persona from a JSON file.

        Args:
            path: Path to the persona file.

        Returns:
            A Persona instance.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Persona file not found: {path}")
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


# Pre-defined sample personas
SAMPLE_PERSONAS = {
    "software_engineer": Persona(
        name="Alex Chen",
        age=28,
        occupation="Senior Software Engineer at a tech startup",
        location="San Francisco, CA",
        living_situation="lives alone in a one-bedroom apartment",
        work_schedule="10am-6pm weekdays, remote with occasional office days",
        habits=[
            "Morning coffee ritual",
            "Daily standup at 10:30am",
            "Lunch break walk",
            "Evening gym session 3x/week",
            "Reading before bed",
        ],
        interests=[
            "Open source projects",
            "Rock climbing",
            "Photography",
            "Cooking Asian cuisine",
            "Podcasts about technology",
        ],
        health_goals=[
            "Exercise regularly",
            "Improve sleep quality",
            "Reduce screen time in evenings",
        ],
        social_circle="Close friends from college, coworkers, climbing gym buddies",
        personality_traits=["introverted", "analytical", "curious", "detail-oriented"],
        typical_day="Wake up around 8am, coffee and news, work from home with breaks "
        "for walks, gym in evening, cook dinner, relax with reading or side projects",
        constraints=["Vegetarian", "Prefers quiet environments"],
    ),
    "graduate_student": Persona(
        name="Maya Rodriguez",
        age=25,
        occupation="PhD student in Cognitive Science",
        location="Boston, MA",
        living_situation="shares apartment with one roommate",
        work_schedule="Flexible, mostly 9am-7pm with irregular hours during deadlines",
        habits=[
            "Morning yoga",
            "Lab work in the afternoon",
            "Evening reading and writing",
            "Weekend brunch with friends",
            "Weekly therapy session",
        ],
        interests=[
            "Neuroscience research",
            "Yoga and meditation",
            "Science fiction novels",
            "Board games",
            "Learning languages (currently: Japanese)",
        ],
        health_goals=[
            "Manage stress better",
            "Maintain work-life balance",
            "Stay active despite long study hours",
        ],
        social_circle="Lab mates, yoga class friends, roommate, long-distance partner",
        personality_traits=["ambitious", "empathetic", "organized", "sometimes anxious"],
        typical_day="Early morning yoga, breakfast, lab work and research, lunch with "
        "colleagues, more research, evening writing or reading, video call with partner",
        constraints=["Budget-conscious", "Needs quiet time for focus"],
    ),
    "freelance_designer": Persona(
        name="Jordan Taylor",
        age=32,
        occupation="Freelance UX/UI Designer",
        location="Austin, TX",
        living_situation="lives with partner and a dog",
        work_schedule="Varies by project, typically 9am-5pm with creative bursts",
        habits=[
            "Morning dog walk",
            "Coffee shop work sessions",
            "Afternoon creative breaks",
            "Evening cooking with partner",
            "Weekend farmers market visits",
        ],
        interests=[
            "Typography and visual design",
            "Urban sketching",
            "Craft cocktails",
            "Live music",
            "Hiking with dog",
        ],
        health_goals=[
            "More consistent exercise",
            "Better posture from desk work",
            "Eat more home-cooked meals",
        ],
        social_circle="Partner, creative community friends, dog park regulars, "
        "online designer community",
        personality_traits=["creative", "extroverted", "spontaneous", "empathetic"],
        typical_day="Morning dog walk, breakfast with partner, work from home or "
        "coffee shop, creative breaks, client calls, evening activities vary",
        constraints=["Needs flexible schedule for creativity", "Dog care responsibilities"],
    ),
}


def get_sample_persona(name: str) -> Persona:
    """Get a pre-defined sample persona.

    Args:
        name: Name of the persona (software_engineer, graduate_student, freelance_designer).

    Returns:
        The requested Persona.

    Raises:
        KeyError: If the persona name is not found.
    """
    if name not in SAMPLE_PERSONAS:
        available = ", ".join(SAMPLE_PERSONAS.keys())
        raise KeyError(f"Unknown persona '{name}'. Available: {available}")
    return SAMPLE_PERSONAS[name]
