from src.utils import COLOR_RED, COLOR_NC

class Dot2SmvNotImplementedError(NotImplementedError):
    def __init__(self, message):            
        # Call the base class constructor with the parameters it needs
        super().__init__(COLOR_RED + message + COLOR_NC)