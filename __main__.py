from data import download, load_data
from preprocessing import dataset_builder, pipeline
from training import model, train
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

if __name__ == "__main__":
#     birds to pull from database
    native_birds = ["American Wigeon",
                    "American Yellow Warbler",
                    "Baltimore Oriole",
                    "Bell's Vireo",
                    "Black-capped Vireo",
                    "Blue-grey Gnatcatcher",
                    "Blue-winged Teal",
                    "Burrowing Owl",
                    "Carolina Chickadee",
                    "Carolina Wren",
                    "Chuck-will's-widow",
                    "Common Grackle",
                    "Red-bellied Woodpecker",
                    "White-eyed Vireo"]
    # for testing
    # native_birds = ["Scissor-tailed Flycatcher"]
    batch_dir = "batch_data"
    birds_to_process = []
    for bird in native_birds:
        batch_file = os.path.join(batch_dir, f"{bird}_batch.npy")
        if not os.path.exists(batch_file):
            birds_to_process.append(bird)
    
    if not birds_to_process:
        print("All bird batches already exist. Skipping batch creation.")
    else:
        getter = download.XenoCantoClient(api_key=api_key)
        all_bird_sounds = download.XenoCantoClient.get_bird_call_list(bird_list=birds_to_process)

        for bird in all_bird_sounds:
            list_of_sounds = all_bird_sounds[bird]
            species_spectrograms = pipeline.get_spectrogram_list(list_of_sounds)

            # for testing
            # convert_mp3_to_spectrogram.display_spectrogram_batch(species_spectrograms)

            pipeline.save_spectrogram_DB(bird, species_spectrograms)

    specs, labels = load_data.load_spectrogram_batches()

    builder = dataset_builder.DatasetBuilder(specs, labels)

    X, y = builder.prepare()
    spec_train, spec_test, labels_train, labels_test = builder.split(X, y)

    model = model.create_model(spec_train.shape[1:], spec_test.shape[1])

    # import numpy as np
    # unique, counts = np.unique(labels_train.argmax(axis=1), return_counts=True)
    # print(dict(zip(unique, counts)))

    train.train_model(model, spec_train, spec_test, labels_train, labels_test)

    # train_model.evaluate_model(spec_test, labels_test)


    
