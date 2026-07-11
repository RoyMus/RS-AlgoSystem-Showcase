import secrets

def generate_api_token(length: int = 64) -> str:
    """Generate a secure random API token."""
    return secrets.token_hex(length // 2)  # token_hex generates 2 chars per byte

if __name__ == "__main__":
    token = generate_api_token()
    print(f"Generated API token: {token}")
