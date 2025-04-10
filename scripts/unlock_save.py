import Metashape

# Open the existing document
doc = Metashape.Document()
doc.open("/home/attilla/photogrammetry/boh-yai/boh-yai.psx")

# Save it (will fail if still locked or opened elsewhere)
doc.save()
