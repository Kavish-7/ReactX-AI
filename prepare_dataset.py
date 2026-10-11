from pathlib import Path
import shutil

DOWNLOADS = Path.home() / "Downloads"
OUTPUT = Path(r"D:\Chemovista\Chemovista_Dataset")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

CLASS_MAPPING = {
    "HDPE": "HDPE",
    "PET": "PET",
    "PP": "PP",
    "LDPE": "OTHER",
    "PS": "OTHER",
}


def find_dataset(keyword):
    """
    Search Downloads and its subfolders for a dataset
    containing train, valid and test folders.
    """

    for folder in DOWNLOADS.rglob("*"):

        if not folder.is_dir():
            continue

        name = folder.name.lower()

        if keyword.lower() not in name:
            continue

        train = folder / "train"
        valid = folder / "valid"
        test = folder / "test"

        if train.is_dir() and valid.is_dir() and test.is_dir():
            return folder

    return None


def copy_images(source, destination, prefix):

    if not source.exists():
        print(f"  WARNING: {source}")
        return 0

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    count = 0

    for image in source.iterdir():

        if (
            image.is_file()
            and image.suffix.lower() in IMAGE_EXTENSIONS
        ):

            new_name = (
                f"{prefix}_{count:05d}"
                f"{image.suffix.lower()}"
            )

            shutil.copy2(
                image,
                destination / new_name
            )

            count += 1

    return count


print()
print("==============================================")
print("       CHEMOVISTA DATASET PREPARATION")
print("==============================================")
print()

print("Searching Downloads...")

MAIN_DATASET = find_dataset("Plastic Sorting Algorithm")
PVC_DATASET = find_dataset("plastic_PVC_2")

print()

if MAIN_DATASET:
    print("MAIN DATASET FOUND:")
    print(MAIN_DATASET)
else:
    print("❌ MAIN DATASET NOT FOUND")

if PVC_DATASET:
    print()
    print("PVC DATASET FOUND:")
    print(PVC_DATASET)
else:
    print()
    print("❌ PVC DATASET NOT FOUND")


if MAIN_DATASET is None or PVC_DATASET is None:

    print()
    print("Please check that both datasets are extracted")
    print("inside your Downloads folder.")

    input("\nPress Enter to exit...")
    raise SystemExit


# ============================================================
# CREATE FINAL DATASET
# ============================================================

for split in ["train", "valid", "test"]:

    print()
    print("----------------------------------------------")
    print(f"PROCESSING {split.upper()}")
    print("----------------------------------------------")

    main_split = MAIN_DATASET / split
    pvc_split = PVC_DATASET / split

    output_split = OUTPUT / split

    # Main dataset
    for source_class, target_class in CLASS_MAPPING.items():

        source = main_split / source_class
        destination = output_split / target_class

        copied = copy_images(
            source,
            destination,
            source_class.lower()
        )

        print(
            f"{source_class:5} -> "
            f"{target_class:5} : "
            f"{copied} images"
        )

    # PVC dataset
    pvc_source = pvc_split / "pvc"
    pvc_destination = output_split / "PVC"

    copied = copy_images(
        pvc_source,
        pvc_destination,
        "pvc"
    )

    print(
        f"{'PVC':5} -> "
        f"{'PVC':5} : "
        f"{copied} images"
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("==============================================")
print("       DATASET CREATION COMPLETE")
print("==============================================")
print()

for split in ["train", "valid", "test"]:

    print(f"{split.upper()}:")

    for class_name in [
        "HDPE",
        "PET",
        "PP",
        "PVC",
        "OTHER"
    ]:

        folder = OUTPUT / split / class_name

        if folder.exists():

            count = sum(
                1
                for f in folder.iterdir()
                if (
                    f.is_file()
                    and f.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            )

        else:
            count = 0

        print(
            f"  {class_name:5}: {count}"
        )

print()
print("FINAL DATASET:")
print(OUTPUT)

input("\nPress Enter to exit...")