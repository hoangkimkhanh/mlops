import os
import argparse
import webbrowser
import requests

base_api = f"http://localhost:8005"

class DisplayImage:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description="arguments")
        parser.add_argument(
            "--save_dir", type=str, help="html file to display", required=True
        )

        query_group = parser.add_mutually_exclusive_group()
        query_group.add_argument("--image_query", type=str, help="image file query")

        args = parser.parse_args()
        print(args)
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.image_query = args.image_query

    def request_image(self, image_file):
        image_api = f"{base_api}/display_image"
        image = open(image_file, "rb")
        files = {"file": image}

        response = requests.post(image_api, files=files)
        if response.status_code == 200:
            self._display(response)

    # @staticmethod
    def _display(self, response):
        with open(f"{self.save_dir}/temp.html", "w", encoding="utf-8") as f:
            f.write(response.text)
            webbrowser.open(f"{self.save_dir}/temp.html")

    def main(self):
        self.request_image(self.image_query)


if __name__ == "__main__":
    DisplayImage().main()