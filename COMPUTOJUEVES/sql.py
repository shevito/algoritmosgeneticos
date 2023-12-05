import sqlite3

# Conectar a la base de datos SQLite (reemplaza 'your_database.db' con el nombre de tu archivo de base de datos)
conn = sqlite3.connect('your_database.db')
cursor = conn.cursor()

# Consultas con una sola tabla:

# Seleccionar todos los artistas
cursor.execute("SELECT * FROM artists;")
artists = cursor.fetchall()
for artist in artists:
    print(artist)

# Seleccionar todas las pistas que sean de un precio específico (por ejemplo, 0.99)
cursor.execute("SELECT * FROM tracks WHERE UnitPrice = 0.99;")
tracks = cursor.fetchall()
for track in tracks:
    print(track)

# Contar el número total de clientes
cursor.execute("SELECT COUNT(*) FROM customers;")
total_customers = cursor.fetchone()
print(total_customers)

# Consultas con operaciones entre dos tablas:

# Seleccionar todos los álbumes de un artista específico (por ejemplo, "The Beatles")
cursor.execute("""
    SELECT albums.Title
    FROM artists
    JOIN albums ON artists.ArtistId = albums.ArtistId
    WHERE artists.Name = 'The Beatles';
""")
albums = cursor.fetchall()
for album in albums:
    print(album)

# Seleccionar todas las pistas de un género específico (por ejemplo, "Rock")
cursor.execute("""
    SELECT tracks.Name
    FROM genres
    JOIN tracks ON genres.GenreId = tracks.GenreId
    WHERE genres.Name = 'Rock';
""")
rock_tracks = cursor.fetchall()
for rock_track in rock_tracks:
    print(rock_track)

# Mostrar el total gastado por cada cliente
cursor.execute("""
    SELECT customers.FirstName, customers.LastName, SUM(invoice_items.UnitPrice * invoice_items.Quantity) as TotalSpent
    FROM customers
    JOIN invoices ON customers.CustomerId = invoices.CustomerId
    JOIN invoice_items ON invoices.InvoiceId = invoice_items.InvoiceId
    GROUP BY customers.CustomerId;
""")
customer_spending = cursor.fetchall()
for spending in customer_spending:
    print(spending)

# Cerrar la conexión
conn.close()
