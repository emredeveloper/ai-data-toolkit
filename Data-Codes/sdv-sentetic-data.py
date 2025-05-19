from sdv.datasets.demo import download_demo
from sdv.single_table import GaussianCopulaSynthesizer

real_data, metadata = download_demo(
    modality='single_table',
    dataset_name='fake_hotel_guests')

print(real_data.head())
print(metadata)
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(data=real_data)

synthetic_data = synthesizer.sample(num_rows=100)
print(synthetic_data.head())