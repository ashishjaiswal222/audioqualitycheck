class AudioVerificationError(Exception):
    """Exception raised for semantic or I/O errors during audio processing."""
    
    def __init__(self, code: str, message: str, status_code: int = 422):
        self.code = code
        self.message = message
        self.status_code = status_code
        # Pass all args to super() so multiprocessing can accurately unpickle it
        super().__init__(code, message, status_code)
