from app import create_app
from app.demo import seed_demo_records


app = create_app()


def seed_demo_data() -> None:
    with app.app_context():
        seed_demo_records()
        print("Demo data is ready.")


if __name__ == "__main__":
    seed_demo_data()
