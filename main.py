import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Import für Datumsformatierung
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import datetime

# Angepasste Funktion zur Kalibrierung des Bildes
def calib_image(l, b, raw_image):
    # Bild kopieren
    raw_image_copy = raw_image.copy()

    # Liste zum Speichern der Punkte
    points = []

    # Funktion zum Erfassen der Mausklicks
    def mouse_handler(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(raw_image_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Klicke auf die vier Ecken des A4-Blattes", raw_image_copy)
            if len(points) == 4:
                cv2.destroyAllWindows()

    # Anzeigen des Bildes und Erfassen der Mausklicks
    cv2.namedWindow("Klicke auf die vier Ecken des A4-Blattes", cv2.WINDOW_NORMAL)
    cv2.imshow("Klicke auf die vier Ecken des A4-Blattes", raw_image_copy)
    cv2.setMouseCallback("Klicke auf die vier Ecken des A4-Blattes", mouse_handler)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 4:
        print("Es wurden nicht genau 4 Punkte ausgewählt.")
        return None, None, None

    # Konvertieren der Punkte in ein NumPy-Array
    pts = np.array(points, dtype='float32')

    # Ordnung der Punkte bestimmen
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]   # Oben links
        rect[2] = pts[np.argmax(s)]   # Unten rechts
        rect[1] = pts[np.argmin(diff)]  # Oben rechts
        rect[3] = pts[np.argmax(diff)]  # Unten links

        return rect

    rect = order_points(pts)

    # Koordinaten der Ecken
    (tl, tr, br, bl) = rect

    # Zielkoordinaten für die transformierten Ecken
    dst = np.array([tl, tr, br, bl], dtype="float32")

    # Berechnung der Transformationsmatrix
    M = cv2.getPerspectiveTransform(rect, dst)

    # Anwendung der Perspektivtransformation auf das gesamte Bild
    warped = cv2.warpPerspective(raw_image, M, (raw_image.shape[1], raw_image.shape[0]))

    # Skalierungsfaktoren berechnen
    pixel_width = np.linalg.norm(tr - tl)
    pixel_height = np.linalg.norm(bl - tl)
    mm_per_pixel_x = l / pixel_width
    mm_per_pixel_y = b / pixel_height

    return warped, mm_per_pixel_x, mm_per_pixel_y

# Funktion zur Detektion der roten Punkte
def detect_red_points(img):
    # Erstellen einer Kopie des Bildes für die Anzeige
    img_display = img.copy()

    # Funktion zur Erfassung des Mausklicks für die Farbauswahl
    def select_color(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param['color_marker'] = (x, y)
            cv2.destroyAllWindows()

    # Anzeige des Bildes und Erfassung der Farbauswahl
    color_param = {}
    cv2.namedWindow('Bild', cv2.WINDOW_NORMAL)
    cv2.imshow('Bild', img_display)
    cv2.setMouseCallback('Bild', select_color, color_param)
    cv2.waitKey(0)

    if 'color_marker' not in color_param:
        print("Kein Punkt ausgewählt.")
        return None, None, None, None

    color_marker = color_param['color_marker']

    # Extrahieren des RGB-Werts des ausgewählten Punkts
    RGB = img[color_marker[1], color_marker[0], :]

    # Konvertieren des Bildes und des ausgewählten Punkts in den HSV-Farbraum
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    selected_color_hsv = cv2.cvtColor(np.uint8([[RGB]]), cv2.COLOR_BGR2HSV)[0][0]

    # Definieren der Schwellenwerte für die Farberkennung
    hue_low = max(selected_color_hsv[0] - 10, 0)
    hue_high = min(selected_color_hsv[0] + 10, 179)
    sat_low = max(selected_color_hsv[1] - 76, 0)
    sat_high = min(selected_color_hsv[1] + 76, 255)
    val_low = max(selected_color_hsv[2] - 76, 0)
    val_high = min(selected_color_hsv[2] + 76, 255)

    # Erstellen der Farbmaske
    lower_bound = np.array([hue_low, sat_low, val_low])
    upper_bound = np.array([hue_high, sat_high, val_high])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Anwenden von morphologischen Operationen
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_filled = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    # Konvertieren der Maske zu einem 8-Bit-Bild
    mask_filled = cv2.convertScaleAbs(mask_filled)

    # Finden der Kreise in der Maske
    circles = cv2.HoughCircles(mask_filled, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=15, minRadius=2, maxRadius=50)

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        centers = circles[:, :2]
        radii = circles[:, 2]

        # Falls mehr als 21 Punkte erkannt wurden, nur die ersten 21 verwenden
        if len(centers) > 21:
            centers = centers[:21]
            radii = radii[:21]
    else:
        centers = np.array([])
        radii = np.array([])

    # Umwandlung in float zur Vermeidung von Überläufen
    centers = centers.astype(np.float64)
    radii = radii.astype(np.float64)

    # Invertieren der y-Koordinaten
    centers[:, 1] = centers[:, 1] * -1
    color_marker = np.array([color_marker[0], color_marker[1] * -1], dtype=np.float64)

    # Finden des ersten Schusses
    if len(centers) > 0:
        distances_to_marker = np.linalg.norm(centers - color_marker, axis=1)
        first_shot_index = np.argmin(distances_to_marker)
    else:
        first_shot_index = None

    # Anzeige der erkannten Punkte
    img_with_circles = img.copy()
    if len(centers) > 0:
        for i, (center, radius) in enumerate(zip(centers, radii)):
            center_plot = (int(round(center[0])), int(round(-center[1])))
            if i == first_shot_index:
                cv2.circle(img_with_circles, center_plot, int(round(radius)), (0, 0, 255), 6)  # Rot für ersten Schuss
            else:
                cv2.circle(img_with_circles, center_plot, int(round(radius)), (255, 255, 0), 6)  # Cyan für andere

    cv2.namedWindow('Erkannte Punkte', cv2.WINDOW_NORMAL)
    cv2.imshow('Erkannte Punkte', img_with_circles)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Benutzerüberprüfung
    answer = messagebox.askquestion("Bestätigung", "Sind die erkannten Punkte korrekt?")
    if answer == 'no':
        # Manuelle Auswahl der Punkte
        points = []

        def select_points(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 21:
                points.append((x, y))
                cv2.circle(param['image'], (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Klicke auf die 21 roten Punkte', param['image'])
                if len(points) == 21:
                    cv2.destroyAllWindows()

        points_param = {'image': img.copy()}
        cv2.namedWindow('Klicke auf die 21 roten Punkte', cv2.WINDOW_NORMAL)
        cv2.imshow('Klicke auf die 21 roten Punkte', points_param['image'])
        cv2.setMouseCallback('Klicke auf die 21 roten Punkte', select_points, points_param)
        cv2.waitKey(0)

        centers = np.array(points, dtype=np.float64)
        radii = np.ones(len(centers)) * 15  # Standardradius
        # Invertieren der y-Koordinaten
        centers[:, 1] = centers[:, 1] * -1
        # Finden des ersten Schusses (erster Klick)
        first_shot_index = 0

    print('Erkennung der roten Punkte abgeschlossen.')

    return centers, radii, color_marker, first_shot_index

# Funktion zur Berechnung der Fläche
def calculate_area(positions):
    # Zentrierung der Daten
    data = positions - np.mean(positions, axis=0)

    # Kovarianzmatrix und Eigenwerte/-vektoren
    covariance = np.cov(data, rowvar=False)
    eigenval, eigenvec = np.linalg.eig(covariance)

    # Berechnung der Fläche
    area = np.pi * np.std(data[:, 0]) * np.std(data[:, 1])

    return area, covariance, data

# Funktion zur Berechnung der Distanz
def calculate_distance(mean_pos):
    distance = np.sqrt(mean_pos[0] ** 2 + mean_pos[1] ** 2)
    return distance

# Funktion zum Speichern der Ergebnisse
def save_results(name, date, discipline, birthday, gender, age, area_mm2, distance):
    # Fläche in cm² umrechnen
    area_cm2 = area_mm2 / 100  # 1 cm² = 100 mm²

    # Prüfen, ob die Excel-Datei bereits existiert
    file_exists = os.path.isfile('Ergebnisse.xlsx')

    if file_exists:
        # Bestehende Daten laden
        df = pd.read_excel('Ergebnisse.xlsx')
    else:
        # Neue DataFrame erstellen
        df = pd.DataFrame(columns=['ID', 'Name', 'Datum', 'Disziplin', 'Geburtstag', 'Geschlecht', 'Alter', 'Fläche', 'Distanz'])

    # Neue ID generieren
    new_id = df['ID'].max() + 1 if not df.empty else 1

    # Neue Datenzeile erstellen
    new_entry = {
        'ID': new_id,
        'Name': name,
        'Datum': date,
        'Disziplin': discipline,
        'Geburtstag': birthday,
        'Geschlecht': gender,
        'Alter': age,
        'Fläche': area_cm2,
        'Distanz': distance
    }

    # Datenzeile hinzufügen
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

    # Daten in Excel-Datei speichern
    df.to_excel('Ergebnisse.xlsx', index=False)

# Funktion zur Erstellung der Verlaufsdiagramme
def create_progress_plots(name):
    # Daten aus Excel-Datei laden
    df = pd.read_excel('Ergebnisse.xlsx')

    # Daten des Athleten filtern
    athlete_data = df[df['Name'] == name]

    if athlete_data.empty:
        print(f"Keine Daten für Athlet {name} gefunden.")
        return

    # Daten nach Datum sortieren
    athlete_data = athlete_data.sort_values(by='Datum')

    # Sicherstellen, dass das Datum im richtigen Format ist
    athlete_data['Datum'] = pd.to_datetime(athlete_data['Datum'], dayfirst=True)

    # Ordner "Verlaufsdaten" erstellen, falls nicht vorhanden
    if not os.path.exists('Verlaufsdaten'):
        os.makedirs('Verlaufsdaten')

    # Fläche über die Zeit plotten
    plt.figure()
    plt.plot(athlete_data['Datum'], athlete_data['Fläche'], marker='o')
    plt.title(f'Flächenverlauf für {name}')
    plt.xlabel('Datum')
    plt.ylabel('Fläche [cm²]')  # Einheit ergänzen

    # Datumsformat anpassen
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join('Verlaufsdaten', f'{name}_Fläche.png'))
    plt.close()

    # Distanz über die Zeit plotten
    plt.figure()
    plt.plot(athlete_data['Datum'], athlete_data['Distanz'], marker='o')
    plt.title(f'Distanzverlauf für {name}')
    plt.xlabel('Datum')
    plt.ylabel('Distanz [mm]')  # Einheit ergänzen

    # Datumsformat anpassen
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join('Verlaufsdaten', f'{name}_Distanz.png'))
    plt.close()

# Funktionen zum Zeichnen der Zielscheiben
def draw_rifle_target(ax):
    # Radien der Ringe in mm
    radii = np.arange(45.5, 0.5, -5)
    for radius in radii:
        if radius < 31:
            circle = plt.Circle((0, 0), radius / 2, facecolor='black', edgecolor='white', linewidth=0.2)
        else:
            circle = plt.Circle((0, 0), radius / 2, facecolor='white', edgecolor='black', linewidth=0.2)
        ax.add_patch(circle)
    ax.set_aspect('equal')

def draw_pistol_target(ax):
    # Radien der Ringe in mm
    radii = np.arange(155.5, 11.5, -16)
    for radius in radii:
        if radius < 60:
            circle = plt.Circle((0, 0), radius / 2, facecolor='black', edgecolor='white', linewidth=0.2)
        else:
            circle = plt.Circle((0, 0), radius / 2, facecolor='white', edgecolor='black', linewidth=0.2)
        ax.add_patch(circle)
    # Innerster Kreis
    circle = plt.Circle((0, 0), 2.5, facecolor='black', edgecolor='white', linewidth=0.2)
    ax.add_patch(circle)
    ax.set_aspect('equal')

# Funktion zum Zeichnen der Konfidenzellipse
def plot_gaussian_ellipse(mean, cov, ax, n_std=2.0, **kwargs):
    from matplotlib.patches import Ellipse

    # Eigenwerte und Eigenvektoren der Kovarianzmatrix
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]

    # Berechnung des Winkels der Hauptachse
    vx, vy = eigenvecs[:,0][0], eigenvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # Halbachsenlängen entsprechend der gewünschten Standardabweichung
    width, height = 2 * n_std * np.sqrt(eigenvals)

    # Erstellen der Ellipse
    ellipse = Ellipse(xy=mean, width=width, height=height,
                      angle=np.degrees(theta), **kwargs)

    ax.add_patch(ellipse)
    return ellipse

# Hauptprogramm
def main():
    # Erstellen des Hauptfensters
    root = tk.Tk()
    root.title("Athletendaten eingeben")

    # Variablen für die Eingabefelder
    name_var = tk.StringVar()
    discipline_var = tk.StringVar(value='Gewehr')  # Standardwert setzen
    gender_var = tk.StringVar(value='Männlich')    # Standardwert setzen
    birthday_var = tk.StringVar()
    testdate_var = tk.StringVar()

    # Funktion zum Abrufen der vorhandenen Athleten
    def get_existing_athletes():
        if os.path.isfile('Ergebnisse.xlsx'):
            df = pd.read_excel('Ergebnisse.xlsx')
            return df['Name'].unique().tolist()
        else:
            return []

    # Funktion zum Aktualisieren der Athletenauswahl
    def update_athlete_selection(event):
        selected_name = name_combo.get()
        if selected_name in existing_athletes:
            # Athletendaten aus Excel laden
            df = pd.read_excel('Ergebnisse.xlsx')
            athlete_data = df[df['Name'] == selected_name].iloc[-1]
            discipline_var.set(athlete_data['Disziplin'])
            gender_var.set(athlete_data['Geschlecht'])
            birthday_var.set(athlete_data['Geburtstag'])
        else:
            # Felder leeren
            discipline_var.set('Gewehr')
            gender_var.set('Männlich')
            birthday_var.set('')

    # Labels und Eingabefelder
    tk.Label(root, text="Name des Athleten/der Athletin:").grid(row=0, column=0, sticky='e')
    existing_athletes = get_existing_athletes()
    name_combo = ttk.Combobox(root, textvariable=name_var, values=existing_athletes)
    name_combo.bind("<<ComboboxSelected>>", update_athlete_selection)
    name_combo.grid(row=0, column=1)

    # Disziplin Auswahl mit Radiobuttons
    tk.Label(root, text="Disziplin:").grid(row=1, column=0, sticky='e')
    discipline_frame = tk.Frame(root)
    discipline_frame.grid(row=1, column=1, sticky='w')
    tk.Radiobutton(discipline_frame, text='Gewehr', variable=discipline_var, value='Gewehr').pack(side='left')
    tk.Radiobutton(discipline_frame, text='Pistole', variable=discipline_var, value='Pistole').pack(side='left')

    # Geschlecht Auswahl mit Radiobuttons
    tk.Label(root, text="Geschlecht:").grid(row=2, column=0, sticky='e')
    gender_frame = tk.Frame(root)
    gender_frame.grid(row=2, column=1, sticky='w')
    tk.Radiobutton(gender_frame, text='Männlich', variable=gender_var, value='Männlich').pack(side='left')
    tk.Radiobutton(gender_frame, text='Weiblich', variable=gender_var, value='Weiblich').pack(side='left')

    tk.Label(root, text="Geburtsdatum (dd.mm.yyyy):").grid(row=3, column=0, sticky='e')
    birthday_entry = tk.Entry(root, textvariable=birthday_var)
    birthday_entry.grid(row=3, column=1)

    tk.Label(root, text="Testdatum (dd.mm.yyyy):").grid(row=4, column=0, sticky='e')
    testdate_entry = tk.Entry(root, textvariable=testdate_var)
    testdate_entry.grid(row=4, column=1)

    # Funktion zur Altersüberprüfung
    def verify_age():
        try:
            birthday = datetime.datetime.strptime(birthday_var.get(), '%d.%m.%Y')
            today = datetime.datetime.today()
            age = today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day))
            confirm = messagebox.askyesno("Altersüberprüfung", f"Der Athlet/die Athletin ist {age} Jahre alt. Ist das korrekt?")
            if confirm:
                proceed_to_image_selection(age)
            else:
                messagebox.showinfo("Hinweis", "Bitte das Geburtsdatum korrigieren.")
        except ValueError:
            messagebox.showerror("Fehler", "Bitte das Geburtsdatum im Format dd.mm.yyyy eingeben.")

    # Funktion zur Bildauswahl und Verarbeitung
    def proceed_to_image_selection(age):
        # Bild auswählen
        image_path = filedialog.askopenfilename(title="Wähle ein Bild aus", filetypes=[("Bilder", "*.jpg;*.jpeg;*.png")])
        if image_path:
            # Bild einlesen
            raw_image = cv2.imread(image_path)

            # Kalibrierung
            warped_image, mm_per_pixel_x, mm_per_pixel_y = calib_image(297, 210, raw_image)

            if warped_image is None:
                messagebox.showerror("Fehler", "Kalibrierung fehlgeschlagen.")
                return

            # Punktdetektion
            centers, radii, color_marker, first_shot_index = detect_red_points(warped_image)

            if centers is None or len(centers) == 0:
                messagebox.showerror("Fehler", "Keine Punkte erkannt.")
                return

            # Umrechnung der Koordinaten in mm
            centers_mm = centers.copy()
            centers_mm[:, 0] = centers_mm[:, 0] * mm_per_pixel_x
            centers_mm[:, 1] = centers_mm[:, 1] * mm_per_pixel_y

            # Positionsberechnung
            first_shot_pos = centers_mm[first_shot_index]
            other_shots = np.delete(centers_mm, first_shot_index, axis=0)

            # Berechnungen
            mean_pos = np.mean(other_shots, axis=0)
            distance = np.linalg.norm(mean_pos - first_shot_pos)
            positions = other_shots - mean_pos
            area_mm2, covariance, data = calculate_area(positions)

            # Ergebnisse speichern
            name = name_var.get()
            date = testdate_var.get()
            discipline = discipline_var.get()
            birthday = birthday_var.get()
            gender = gender_var.get()

            # Fläche in cm² speichern
            save_results(name, date, discipline, birthday, gender, age, area_mm2, distance)

            # Verlaufsdiagramme erstellen
            create_progress_plots(name)

            # Ordner "Schlüsselbilder Blindanschlag" erstellen, falls nicht vorhanden
            if not os.path.exists('Schlüsselbilder Blindanschlag'):
                os.makedirs('Schlüsselbilder Blindanschlag')

            # Speicherort bestimmen
            image_filename = f"{name}_{date.replace('.', '-')}.png"
            save_path = os.path.join('Schlüsselbilder Blindanschlag', image_filename)

            # Neuen Code einfügen:
            # Prüfen, ob bereits eine Datei mit diesem Namen existiert und ggf. hochzählen
            if os.path.exists(save_path):
                base_name, ext = os.path.splitext(image_filename)
                counter = 1
                # Schleife so lange, bis ein nicht-existierender Dateiname gefunden ist
                while True:
                    new_filename = f"{base_name}_{counter}{ext}"
                    new_path = os.path.join('Schlüsselbilder Blindanschlag', new_filename)
                    if not os.path.exists(new_path):
                        save_path = new_path
                        break
                    counter += 1

            # Visualisierung erstellen
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Angepasste Größe

            # Subplot 1: Zielscheibe und Datenpunkte
            if discipline == "Gewehr":
                draw_rifle_target(axs[0])
            elif discipline == "Pistole":
                draw_pistol_target(axs[0])

            # Farben definieren
            first_shot_color = 'red'
            other_shots_color = 'cyan'  # Entspricht der Farbe Cyan

            # Punkte plotten
            axs[0].scatter(other_shots[:,0], other_shots[:,1], c=other_shots_color, label='Weitere Schüsse')
            axs[0].scatter(first_shot_pos[0], first_shot_pos[1], c=first_shot_color, label='Erster Schuss')
            axs[0].scatter(mean_pos[0], mean_pos[1], c='blue', marker='x', label='Mittelpunkt')
            axs[0].plot([first_shot_pos[0], mean_pos[0]], [first_shot_pos[1], mean_pos[1]], 'r-')

            # Konfidenzellipse plotten
            plot_gaussian_ellipse(mean_pos, covariance, axs[0], edgecolor='blue', facecolor='none', linewidth=2, label='Konfidenzellipse')

            # Achsen anpassen
            all_x = np.concatenate((other_shots[:, 0], [first_shot_pos[0], mean_pos[0]]))
            all_y = np.concatenate((other_shots[:, 1], [first_shot_pos[1], mean_pos[1]]))
            margin = 10  # Margin in mm

            min_x = np.min(all_x) - margin
            max_x = np.max(all_x) + margin
            min_y = np.min(all_y) - margin
            max_y = np.max(all_y) + margin

            axs[0].set_xlim(min_x, max_x)
            axs[0].set_ylim(min_y, max_y)

            axs[0].set_xlabel('mm')
            axs[0].set_ylabel('mm')
            axs[0].legend()

            # Subplot 2: Bild mit erkannten Kreisen
            axs[1].imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
            # Invertieren der y-Koordinaten für die Darstellung
            centers_plot = centers.copy()
            centers_plot[:,1] = -centers_plot[:,1]

            import matplotlib.patches as patches

            for i, (center, radius) in enumerate(zip(centers_plot, radii)):
                if i == first_shot_index:
                    circle = patches.Circle((center[0], center[1]), radius, edgecolor='red', fill=False, linewidth=2)
                else:
                    circle = patches.Circle((center[0], center[1]), radius, edgecolor='cyan', fill=False, linewidth=2)
                axs[1].add_patch(circle)
            axs[1].axis('off')

            # Anpassung des Titels
            area_cm2 = area_mm2 / 100  # mm² in cm² umrechnen
            fig.suptitle(f"{name}, Fläche: {round(area_cm2, 3)} cm², Distanz: {round(distance, 2)} mm")

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)

            messagebox.showinfo("Erfolg", f"Die Ergebnisse wurden erfolgreich gespeichert.\nBild gespeichert unter: {save_path}")

        else:
            messagebox.showwarning("Hinweis", "Kein Bild ausgewählt.")

    # Button zum Bestätigen und Weiterfahren
    tk.Button(root, text="Weiter", command=verify_age).grid(row=5, column=0, columnspan=2, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
