from enum import Enum

class RemoteFlexibility(Enum):
    REMOTE = "Remote"
    HYBRID = "Hybrid"
    ON_SITE = "On-site"

    def __str__(self):
        return self.value   