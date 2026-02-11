import py4vasp

calc = py4vasp.Calculation.from_path(".")
ph = calc.phonon_band  # <-- questo è l'oggetto descritto nella doc

# salva direttamente un'immagine (pdf/png ecc.)
ph.to_image(filename="LiF_phonons_py4vasp.pdf")
ph.to_image(filename="LiF_phonons_py4vasp.png")
