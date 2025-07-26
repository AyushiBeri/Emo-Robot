import os
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from tkinter import Tk, Button, messagebox, simpledialog
import webbrowser

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


# === Nutrition Data Store ===
def get_today_data():
    csv_path = "/home/ayushi/detected_meals.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded data from CSV: {csv_path}")
            return df
        except Exception as e:
            print(f"❌ Failed to load CSV. Error: {e}")

    # Fallback data
    print("⚠️ CSV not found or failed to load. Using default data.")
    data = {
        'Dish': ['Rice', 'Dal', 'Chapati'],
        'Calories': [200, 150, 120],
        'Protein (g)': [4.5, 6.0, 3.0],
        'Fat (g)': [1.0, 3.5, 2.0],
        'Carbs (g)': [45, 25, 20]
    }
    return pd.DataFrame(data)


# === Generate PDF Report ===
def generate_pdf_report(df, pdf_path):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(30, 750, f"Daily Nutrition Report - {datetime.now().strftime('%Y-%m-%d')}")

    table_y = 700
    c.setFont("Helvetica-Bold", 10)
    for i, col in enumerate(df.columns):
        c.drawString(50 + i * 100, table_y, str(col))

    c.setFont("Helvetica", 10)
    for row in df.itertuples(index=False):
        table_y -= 20
        for i, val in enumerate(row):
            c.drawString(50 + i * 100, table_y, str(val))

    c.save()


# === Upload PDF to Drive with user access ===
def upload_to_drive(file_path, user_emails):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    file = drive.CreateFile({'title': os.path.basename(file_path)})
    file.SetContentFile(file_path)
    file.Upload()

    for email in user_emails:
        try:
            file.InsertPermission({
                'type': 'user',
                'value': email,
                'role': 'reader'
            })
        except Exception as e:
            print(f"❌ Failed to share with {email}: {e}")

    return file['alternateLink']


# === GUI ===
def show_gui(pdf_path, download_url):
    def open_pdf():
        webbrowser.open(f"file://{os.path.abspath(pdf_path)}")

    def open_drive():
        webbrowser.open(download_url)

    root = Tk()
    root.title("Daily Nutrition Report")
    root.geometry("320x180")

    Button(root, text="📥 Open Local PDF", command=open_pdf, width=30).pack(pady=10)
    Button(root, text="☁️ View on Google Drive", command=open_drive, width=30).pack(pady=10)
    Button(root, text="❌ Exit", command=root.destroy, width=30).pack(pady=10)

    root.mainloop()


# === Ask User for Email Access ===
def get_emails_from_user():
    root = Tk()
    root.withdraw()
    email_input = simpledialog.askstring("Access Control", "Enter email IDs to give access (comma-separated):")
    root.destroy()
    if email_input:
        return [email.strip() for email in email_input.split(",")]
    return []


# === Main ===
def main():
    print("📊 Generating today's report...")

    df = get_today_data()
    today_str = datetime.now().strftime("%Y-%m-%d")
    pdf_path = f"nutrition_report_{today_str}.pdf"

    generate_pdf_report(df, pdf_path)
    print("✅ PDF created:", pdf_path)

    emails = get_emails_from_user()
    if not emails:
        print("⚠️ No email provided. File will not be shared with specific users.")
        return

    print("☁️ Uploading to Google Drive and sharing...")
    drive_link = upload_to_drive(pdf_path, emails)
    print("✅ Uploaded to:", drive_link)

    show_gui(pdf_path, drive_link)


if __name__ == "__main__":
    main()
