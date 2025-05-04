class AppException(Exception):
    def __init__(self, message: str, status_code: int = 500, error_type: str = "AppException", details: dict = None):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.details = details or {}
        super().__init__(message)

class AuthenticationException(AppException):
    def __init__(self, message="Authentication failed", details=None):
        super().__init__(message, 401, "AuthenticationException", details)

class ValidationException(AppException):
    def __init__(self, message="Validation error", details=None):
        super().__init__(message, 422, "ValidationException", details)

class ResourceNotFoundException(AppException):
    def __init__(self, message="Resource not found", details=None):
        super().__init__(message, 404, "ResourceNotFoundException", details)

class ExternalServiceException(AppException):
    def __init__(self, message="External service error", details=None):
        super().__init__(message, 503, "ExternalServiceException", details) 