import os
from PIL import Image, ImageDraw, ImageFont

def get_font(font_name, size):
    try:
        # Standard paths for Windows fonts
        paths = [
            f"C:\\Windows\\Fonts\\{font_name}.ttf",
            f"C:\\Windows\\Fonts\\{font_name}b.ttf",
            font_name
        ]
        for p in paths:
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    except Exception:
        pass
    return ImageFont.load_default()

def create_image():
    # 1. Image Settings
    width = 1200
    height = 630
    
    # Colors
    bg_color = (11, 15, 25)          # #0B0F19 (Deep slate)
    card_bg = (22, 31, 48)           # #161F30 (Slightly lighter slate)
    title_color = (255, 255, 255)    # #FFFFFF
    subtitle_color = (148, 163, 184) # #94A3B8
    text_muted = (148, 163, 184)     # #94A3B8
    
    before_accent = (239, 68, 68)    # #EF4444 (Coral Red)
    before_badge_bg = (63, 18, 18)   # Dark Red
    
    after_accent = (16, 185, 129)    # #10B981 (Emerald Green)
    after_badge_bg = (6, 78, 59)     # Dark Green
    
    # Fonts
    title_font = get_font("segoeui", 34)
    subtitle_font = get_font("segoeui", 16)
    section_font = get_font("segoeui", 22)
    label_font = get_font("segoeui", 14)
    score_font = get_font("consolas", 18)
    snippet_font = get_font("segoeui", 14)
    path_font = get_font("segoeui", 13)
    
    # Create image
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # 2. Header
    draw.text((60, 40), "Fixing a Silent Data-Loss Bug in Retryv's Embedding Pipeline", fill=title_color, font=title_font)
    draw.text((60, 95), "Why gemini-embedding-2 on v1beta returned 1 embedding instead of a batch of 100", fill=subtitle_color, font=subtitle_font)
    
    # Divider
    draw.line([(60, 130), (1140, 130)], fill=(30, 41, 59), width=1)
    
    # 3. Columns Setup
    col_width = 510
    left_x = 60
    right_x = 630
    
    # Query Line (shared logic but drawn in both columns)
    def draw_column_header(x_start, label, accent_color, query_text):
        # Column title
        draw.text((x_start, 150), label, fill=accent_color, font=section_font)
        # Query box background
        draw.rounded_rectangle(
            [x_start, 195, x_start + col_width, 235],
            radius=6,
            fill=(30, 41, 59)
        )
        draw.text((x_start + 15, 205), f'Query: "{query_text}"', fill=(226, 232, 240), font=label_font)

    draw_column_header(left_x, "BEFORE: 95% Chunk Data Loss", before_accent, "lifespan events")
    draw_column_header(right_x, "AFTER: Correct Batch Embedding", after_accent, "lifespan events")
    
    # 4. Results Data
    before_results = [
        {"score": "0.0164", "file": "docs/en/docs/alternatives.md", "text": "Flask alternatives comparison and history..."},
        {"score": "0.0161", "file": "docs/en/docs/alternatives.md", "text": "Have sensible defaults, but powerful features..."},
        {"score": "0.0159", "file": "docs/en/docs/alternatives.md", "text": "Requests alternatives details and comparison..."}
    ]
    
    after_results = [
        {"score": "0.0318", "file": "docs/en/docs/advanced/events.md", "text": "# Lifespan Events - define startup & shutdown logic..."},
        {"score": "0.0317", "file": "docs/en/docs/advanced/events.md", "text": "🚨 Keep in mind that these lifespan events will only run..."},
        {"score": "0.0313", "file": "docs/en/docs/advanced/events.md", "text": "new lifespan async context manager parameter..."}
    ]
    
    def draw_results(x_start, results, accent_color, badge_bg_color):
        y_start = 255
        card_height = 85
        card_gap = 18
        
        for i, res in enumerate(results):
            card_y = y_start + i * (card_height + card_gap)
            
            # Draw Card Background
            draw.rounded_rectangle(
                [x_start, card_y, x_start + col_width, card_y + card_height],
                radius=8,
                fill=card_bg,
                outline=(51, 65, 85),
                width=1
            )
            
            # Draw Score Badge
            badge_x1 = x_start + 12
            badge_y1 = card_y + 12
            badge_x2 = x_start + 97
            badge_y2 = card_y + card_height - 12
            
            draw.rounded_rectangle(
                [badge_x1, badge_y1, badge_x2, badge_y2],
                radius=4,
                fill=badge_bg_color
            )
            
            # Score Text (Centered in badge)
            score_text = res["score"]
            # Approximate centering
            draw.text((badge_x1 + 14, badge_y1 + 16), score_text, fill=accent_color, font=score_font)
            
            # Source File
            draw.text((x_start + 112, card_y + 16), res["file"], fill=(248, 250, 252), font=path_font)
            
            # Snippet Text
            draw.text((x_start + 112, card_y + 44), res["text"], fill=text_muted, font=snippet_font)
            
    draw_results(left_x, before_results, before_accent, before_badge_bg)
    draw_results(right_x, after_results, after_accent, after_badge_bg)
    
    # 5. Footer / Branding
    draw.text((60, 580), "Retryv RAG Pipeline Verification System", fill=(71, 85, 105), font=subtitle_font)
    
    # Ensure directory exists
    os.makedirs("reports", exist_ok=True)
    
    # Save Image
    output_path = "reports/linkedin_before_after_lifespan_events.png"
    img.save(output_path, "PNG")
    print(f"Image successfully saved to: {output_path}")

if __name__ == "__main__":
    create_image()
