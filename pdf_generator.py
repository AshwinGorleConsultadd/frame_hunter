"""
PDF Report Generator for Defect Analysis
Generates professional PDF reports with embedded images
"""

import os
import logging
from typing import List, Any, Dict
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_pdf_report(defects_with_images: List[Dict[str, Any]], output_file_path: str) -> bool:
    """
    Generate a professional PDF report with embedded images
    
    Args:
        defects_with_images: List of dictionaries containing defect data with image_path
        output_file_path: Path where the PDF should be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Generating PDF report with %d defects...", len(defects_with_images))
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_file_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        # header_style = ParagraphStyle(
        #     'CustomHeader',
        #     parent=styles['Normal'],
        #     fontSize=10,
        #     fontName='Helvetica-Bold',
        #     alignment=TA_CENTER
        # )
        
        cell_style = ParagraphStyle(
            'CustomCell',
            parent=styles['Normal'],
            fontSize=8,
            fontName='Helvetica',
            alignment=TA_LEFT
        )
        
        # Build the story (content)
        story = []
        
        # Add title
        title = Paragraph("Defect Analysis Report", title_style)
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Add summary
        total_defects = len(defects_with_images)
        defects_with_screenshots = sum(1 for d in defects_with_images if d.get('image_path'))
        summary_text = f"Total Defects Found: {total_defects} | Screenshots Captured: {defects_with_screenshots}"
        summary = Paragraph(summary_text, styles['Normal'])
        story.append(summary)
        story.append(Spacer(1, 20))
        
        # Prepare table data
        table_data = []
        
        # Add headers
        headers = [
            'Building',
            'Apartment', 
            'Tread Number',
            'Priority',
            'Timestamp (s)',
            'Description',
            'Screenshot'
        ]
        table_data.append(headers)
        
        # Add defect data
        for defect in defects_with_images:
            # Get image path from the defect dictionary
            image_path = defect.get('image_path')
            
            # Check if image exists
            if image_path and os.path.exists(image_path):
                try:
                    # Create image object
                    img = Image(image_path, width=1.2*inch, height=0.9*inch)
                    img.hAlign = 'CENTER'
                except Exception as e:
                    logger.warning("Could not load image %s: %s", image_path, e)
                    img = Paragraph("Image<br/>Not Found", cell_style)
            else:
                img = Paragraph("No Image", cell_style)
            
            # Create row data
            row_data = [
                Paragraph(str(defect.get('building_counter') or "N/A"), cell_style),
                Paragraph(str(defect.get('apartment_number') or "N/A"), cell_style),
                Paragraph(str(defect.get('tread_number') or "N/A"), cell_style),
                Paragraph(str(defect.get('priority') or "N/A"), cell_style),
                Paragraph(f"{defect.get('ss_timestamp', 0):.2f}" if defect.get('ss_timestamp') else "N/A", cell_style),
                Paragraph(str(defect.get('description') or "N/A")[:50] + "..." if len(str(defect.get('description') or "")) > 50 else str(defect.get('description') or "N/A"), cell_style),
                img
            ]
            
            table_data.append(row_data)
        
        # Create table
        table = Table(
            table_data,
            colWidths=[
                0.8*inch,  # Building
                0.8*inch,  # Apartment
                0.8*inch,  # Tread Number
                0.6*inch,  # Priority
                0.8*inch,  # Timestamp
                2.0*inch,  # Description
                1.2*inch   # Screenshot
            ],
            repeatRows=1  # Repeat header on each page
        )
        
        # Apply table style
        table_style = TableStyle([
            # Header row styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            
            # Data rows styling
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            
            # Column-specific alignments
            ('ALIGN', (0, 1), (0, -1), 'CENTER'),  # Building
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),  # Apartment
            ('ALIGN', (2, 1), (2, -1), 'CENTER'),  # Tread Number
            ('ALIGN', (3, 1), (3, -1), 'CENTER'),  # Priority
            ('ALIGN', (4, 1), (4, -1), 'CENTER'),  # Timestamp
            ('ALIGN', (5, 1), (5, -1), 'LEFT'),    # Description
            ('ALIGN', (6, 1), (6, -1), 'CENTER'),  # Screenshot
        ])
        
        table.setStyle(table_style)
        story.append(table)
        
        # Add footer
        story.append(Spacer(1, 20))
        footer_text = f"Report generated on {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        footer = Paragraph(footer_text, styles['Normal'])
        story.append(footer)
        
        # Build PDF
        doc.build(story)
        
        logger.info("✅ PDF report generated successfully: %s", output_file_path)
        return True
        
    except Exception as e:
        logger.error("❌ Failed to generate PDF report: %s", e)
        return False

def generate_simple_pdf_report(defects_with_images: List[Dict[str, Any]], output_file_path: str) -> bool:
    """
    Generate a simple PDF report without images (fallback option)
    
    Args:
        defects_with_images: List of dictionaries containing defect data with image_path
        output_file_path: Path where the PDF should be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Generating simple PDF report with %d defects...", len(defects_with_images))
        
        doc = SimpleDocTemplate(output_file_path, pagesize=A4)
        story = []
        
        # Add title
        title = Paragraph("Defect Analysis Report", getSampleStyleSheet()['Title'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Create table data
        table_data = [['Building', 'Apartment', 'Tread', 'Priority', 'Timestamp', 'Description', 'Image Path']]
        
        for defect in defects_with_images:
            image_path = defect.get('image_path', "N/A")
            if image_path == "N/A":
                image_path = defect.get('image_filename', "N/A")
            
            table_data.append([
                str(defect.get('building_counter') or "N/A"),
                str(defect.get('apartment_number') or "N/A"),
                str(defect.get('tread_number') or "N/A"),
                str(defect.get('priority') or "N/A"),
                f"{defect.get('ss_timestamp', 0):.2f}" if defect.get('ss_timestamp') else "N/A",
                str(defect.get('description') or "N/A")[:30] + "..." if len(str(defect.get('description') or "")) > 30 else str(defect.get('description') or "N/A"),
                image_path
            ])
        
        # Create table
        table = Table(table_data, colWidths=[1*inch, 1*inch, 0.8*inch, 0.8*inch, 1*inch, 2*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        doc.build(story)
        
        logger.info("✅ Simple PDF report generated successfully: %s", output_file_path)
        return True
        
    except Exception as e:
        logger.error("❌ Failed to generate simple PDF report: %s", e)
        return False

if __name__ == "__main__":
    # Test the function
    print("PDF Generator module loaded successfully!")
    print("Use generate_pdf_report(defects, output_path) to create reports")
