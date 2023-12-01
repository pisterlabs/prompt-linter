from concurrent.futures import ThreadPoolExecutor, as_completed
from openai_utilts import generate_image

def process_item(item):
    try:
        url = generate_image(item, "1024x1024")
        print(f"Processed: {item}")
        return item, url
    except Exception as e:
        print(f"Error processing {item}: {e}")
        return item, None

def multigen(data):
    """Generate multiple images from a dict of menu items and price.

    Args:
        data (dict): { menu_item: price }

    Returns:
        dict: Dictionary of generated image URLs { item_name: url }
    """
    test_data = data
    result_urls = {}

    with ThreadPoolExecutor() as executor:
        # Process only the first 5 items
        items_to_process = list(test_data.keys())[:5]

        futures = [executor.submit(process_item, item)
                   for item in items_to_process]

        for future in as_completed(futures):
            try:
                item, url = future.result()
                if url:
                    result_urls[item] = url
            except Exception as e:
                print(f"Exception while processing: {e}")

    return result_urls

 # Example usage:
# data = {"biriyani": 10, "alfaham": 15, "shavarma": 20, "rice meals": 25, "choclate cake": 30}
# result_urls = multigen(data)
