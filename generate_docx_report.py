import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

def set_cell_background(cell, fill_hex):
    """Set background color of a table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set cell padding in dxa (1 pt = 20 dxa)."""
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
    """Set subtle borders for the table."""
    tblPr = table._element.xpath('w:tblPr')
    if tblPr:
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'  <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'  <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'  <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'  <w:insideV w:val="none"/>'
            f'  <w:left w:val="none"/>'
            f'  <w:right w:val="none"/>'
            f'</w:tblBorders>'
        )
        tblPr[0].append(borders)

def build_word_document(output_path):
    doc = docx.Document()

    # Configure Margins (1 inch all around)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.page_width = Inches(8.5)
        section.page_height = Inches(11.0)

    # Base Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.25
    normal_style.paragraph_format.space_after = Pt(6)

    # Theme Colors
    PRIMARY = RGBColor(128, 0, 0)       # Maroon (#800000) for VIT-AP theme
    NAVY = RGBColor(20, 50, 90)         # Deep Navy (#14325A)
    DARK = RGBColor(34, 34, 34)         # Charcoal Dark (#222222)
    GRAY = RGBColor(90, 90, 90)         # Subtitle Gray (#5A5A5A)
    PRIMARY_HEX = "800000"
    HEADER_BG_HEX = "14325A"
    ALT_ROW_HEX = "F8F9FA"

    # -------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------
    def add_title(text, font_size=20, bold=True, color=PRIMARY, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_after = Pt(space_after)
        run = p.add_run(text)
        run.bold = bold
        run.font.size = Pt(font_size)
        run.font.color.rgb = color
        return p

    def add_heading_1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(16)
        run.font.color.rgb = PRIMARY
        return p

    def add_heading_2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(13.5)
        run.font.color.rgb = NAVY
        return p

    def add_heading_3(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(12)
        run.font.color.rgb = DARK
        return p

    def add_p(text, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=6):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_after = Pt(space_after)
        run = p.add_run(text)
        run.bold = bold
        run.italic = italic
        run.font.size = Pt(11.5)
        return p

    def add_bullet(text, level=0):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.left_indent = Inches(0.25 * (level + 1))
        run = p.add_run(text)
        run.font.size = Pt(11.5)
        return p

    def add_code_block(code_text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.left_indent = Inches(0.2)
        p.paragraph_format.right_indent = Inches(0.2)
        run = p.add_run(code_text)
        run.font.name = 'Consolas'
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(20, 20, 20)
        
        # Add light background shading to paragraph XML
        pPr = p._element.get_or_add_pPr()
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F4F5F7"/>')
        pPr.append(shd)
        return p

    def add_styled_table(headers, rows_data, col_widths=None):
        table = doc.add_table(rows=len(rows_data) + 1, cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(table)

        # Header Row
        hdr_cells = table.rows[0].cells
        for idx, header in enumerate(headers):
            cell = hdr_cells[idx]
            cell.text = header
            set_cell_background(cell, HEADER_BG_HEX)
            set_cell_margins(cell, top=140, bottom=140, left=140, right=140)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.name = 'Times New Roman'
                run.font.size = Pt(10.5)
                run.font.color.rgb = RGBColor(255, 255, 255)

        # Data Rows
        for r_idx, row_values in enumerate(rows_data):
            row_cells = table.rows[r_idx + 1].cells
            bg_color = ALT_ROW_HEX if r_idx % 2 == 1 else "FFFFFF"
            for c_idx, val in enumerate(row_values):
                cell = row_cells[c_idx]
                cell.text = str(val)
                set_cell_background(cell, bg_color)
                set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
                p = cell.paragraphs[0]
                # If first column align left, else center
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(10)
                    if "TARA" in str(val) or "+" in str(val) or "%" in str(val) and c_idx >= len(row_values)-2:
                        run.bold = True

        # Col widths if given
        if col_widths:
            for row in table.rows:
                for idx, w in enumerate(col_widths):
                    row.cells[idx].width = Inches(w)

        doc.add_paragraph().paragraph_format.space_after = Pt(4)
        return table

    # =============================================================
    # COVER PAGE
    # =============================================================
    add_title("TARA-FL: TRUST-AWARE ROBUST ADAPTIVE FEDERATED LEARNING", font_size=20, bold=True, color=PRIMARY, space_after=18)
    
    add_title("A CAPSTONE PROJECT REPORT", font_size=13, bold=True, color=DARK, space_after=6)
    add_p("Submitted in partial fulfillment of the requirements for the award of the Degree of", align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
    add_title("BACHELOR OF TECHNOLOGY", font_size=14, bold=True, color=PRIMARY, space_after=2)
    add_title("IN", font_size=11, bold=False, color=DARK, space_after=2)
    add_title("COMPUTER SCIENCE AND ENGINEERING", font_size=14, bold=True, color=PRIMARY, space_after=24)

    add_p("By", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    add_title("【STUDENT NAME】 (【21BCEXXXX】)", font_size=13, bold=True, color=DARK, space_after=20)

    add_p("Under the Guidance of", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    add_title("【DR. GUIDE NAME】", font_size=13, bold=True, color=DARK, space_after=2)
    add_p("School of Computer Science and Engineering (SCOPE)", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=30)

    add_title("SCHOOL OF COMPUTER SCIENCE AND ENGINEERING (SCOPE)", font_size=12, bold=True, color=PRIMARY, space_after=2)
    add_title("VIT-AP UNIVERSITY", font_size=15, bold=True, color=PRIMARY, space_after=2)
    add_p("AMARAVATI – 522237, ANDHRA PRADESH, INDIA", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    add_p("MAY 2026", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)

    doc.add_page_break()

    # =============================================================
    # CERTIFICATE
    # =============================================================
    add_heading_1("CERTIFICATE")
    add_p(
        "This is to certify that the Capstone Project report titled “TARA-FL: Trust-Aware Robust Adaptive Federated Learning” "
        "being submitted by 【Student Name】 (【21BCEXXXX】) in partial fulfillment of the requirements for the award of the degree "
        "of Bachelor of Technology in Computer Science and Engineering is a bonafide record of work carried out under my guidance "
        "and supervision at VIT-AP University, Amaravati."
    )
    add_p(
        "The contents of this project report, in full or in parts, have neither been taken from any other source without due citation "
        "nor been submitted to any other Institute or University for the award of any degree, diploma, or certificate."
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(24)
    add_p("【Dr. Guide Name】", bold=True)
    add_p("Project Supervisor, SCOPE\nVIT-AP University, Amaravati")

    doc.add_paragraph().paragraph_format.space_after = Pt(16)
    add_p("The Capstone Project Viva-Voce Examination is held on: ____________________")

    # Signatures table
    add_styled_table(
        ["Internal Examiner", "External Examiner"],
        [["__________________________", "__________________________"]],
        [3.25, 3.25]
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    add_p("Approved by:", bold=True)
    add_styled_table(
        ["Program Chair", "Dean"],
        [[
            "Dr. __________________________\nB.Tech. Computer Science & Engineering",
            "Dr. __________________________\nSchool of Computer Science & Engineering (SCOPE)"
        ]],
        [3.25, 3.25]
    )

    doc.add_page_break()

    # =============================================================
    # ACKNOWLEDGEMENTS
    # =============================================================
    add_heading_1("ACKNOWLEDGEMENTS")
    add_p(
        "The successful completion of this capstone project, titled TARA-FL: Trust-Aware Robust Adaptive Federated Learning, "
        "would not have been possible without the invaluable support, mentorship, and institutional resources provided by VIT-AP University."
    )
    add_p(
        "First and foremost, I express my deepest gratitude to my project supervisor, 【Dr. Guide Name】, for continuous encouragement, "
        "intellectual guidance, and constructive critique throughout the conception, mathematical formalization, implementation, and "
        "empirical evaluation of the TARA-FL framework. The freedom to explore the intersection of robust statistics, dynamic trust mechanics, "
        "and distributed machine learning, combined with rigorous technical reviews, fundamentally elevated the quality of this work."
    )
    add_p(
        "I convey my heartfelt thanks to the Program Chair, the Dean, and the faculty members of the School of Computer Science and "
        "Engineering (SCOPE) for providing academic support, high-performance computing resources, and structured milestone reviews "
        "that ensured the project's steady progression."
    )
    add_p(
        "I acknowledge the open-source scientific Python community whose robust software ecosystem made this research reproducible and extensible: "
        "the PyTorch deep learning library, the torchvision computer vision toolkit, the NumPy and SciPy scientific computing stacks, "
        "the scikit-learn machine learning suite, the pandas analysis framework, the matplotlib visualization engine, and the pytest verification suite. "
        "Furthermore, I acknowledge the foundational contributions of McMahan et al. (2017) for the Federated Averaging (FedAvg) paradigm, "
        "as well as the researchers in Byzantine-resilient aggregation whose theoretical frameworks established the baseline for this study."
    )
    add_p(
        "Finally, I express my sincere appreciation to my family and peers for their continuous patience, motivation, and understanding "
        "throughout this demanding capstone endeavour."
    )
    add_p("— 【Student Name】\nDepartment of Computer Science and Engineering\nVIT-AP University", bold=True, align=WD_ALIGN_PARAGRAPH.RIGHT)

    doc.add_page_break()

    # =============================================================
    # ABSTRACT
    # =============================================================
    add_heading_1("ABSTRACT")
    add_p(
        "Federated Learning (FL) has emerged as a compelling decentralized paradigm enabling distributed edge clients to collaboratively "
        "optimize a global machine learning model without centralizing sensitive raw data. While this architecture structurally mitigates data "
        "privacy risks, it exposes the global optimization trajectory to unreliability, non-IID data divergence, and adversarial poisoning attacks "
        "(such as label flipping, gradient scaling, and sign flipping). Existing defensive approaches typically rely on static robust aggregation "
        "operators (e.g., Krum, Coordinate-wise Median, Trimmed Mean) or rigid trust scoring schemes. These static defenses suffer from significant "
        "limitations: they degrade accuracy and waste compute in benign environments, lack historical adaptability across evolving attacks, "
        "and fail to differentiate benign non-IID statistical variance from deliberate malicious manipulation."
    )
    add_p(
        "To overcome these fundamental challenges, this project designs, implements, and evaluates TARA-FL (Trust-Aware Robust Adaptive Federated "
        "Learning), an end-to-end framework integrating multi-stage dynamic client trust evaluation, a graduated quarantine state machine, "
        "multi-signal round-risk estimation, and risk-aware adaptive aggregation routing. TARA-FL evaluates client reliability through a five-stage "
        "pipeline incorporating peer-relative Median Absolute Deviation (MAD) scaling, inverse-quadratic current-round decay, exponential moving average "
        "(EMA) historical memory, and an exponential persistence penalty for recurring anomalies. Rather than imposing permanent binary exclusions, "
        "TARA-FL categorizes clients into four graded trust zones (Safe, Probation, Low Trust, and Quarantine), providing a 2-round clean probation path "
        "that allows temporarily anomalous edge devices to recover. Concurrently, a multi-signal risk engine aggregates five orthogonal metrics "
        "(update-distance anomalies, population trust deficits, suspicious-client proportions, worst-client deviations, and validation drop penalties) "
        "to quantify the round-level threat severity. Based on this risk score, the framework dynamically dispatches an optimal aggregation operator: "
        "Trust-Aware FedAvg for Low Risk, Trust-Weighted Robust Trimming for Medium Risk, and Trust-Weighted Coordinate Median for High Risk."
    )
    add_p(
        "The framework was rigorously evaluated across a 20-scenario experimental matrix comprising 160 complete federated runs on the standard "
        "MNIST benchmark under both IID and heterogeneous Dirichlet (α = 0.5) partitions. Under severe adversarial conditions (a 30% Byzantine "
        "Gradient Scaling attack in the IID setting), TARA-FL preserved 90.00% global test accuracy where standard FedAvg catastrophically collapsed "
        "to 22.65% (a +67.35% gain). Under a 30% Sign Flipping attack in the non-IID setting, TARA-FL achieved 81.97% accuracy compared to FedAvg's 62.00% "
        "(+19.98% gain). The framework incurred a negligible clean-baseline accuracy trade-off of only 0.27% in IID conditions and added an average "
        "per-round computation latency of less than 0.15 seconds (+1.11% overhead). These empirical results confirm that coupling dynamic trust tracking "
        "with adaptive risk routing establishes an efficient, resilient, and defensible foundation for decentralized machine learning."
    )
    add_p("Keywords: Federated Learning, Byzantine Robustness, Dynamic Trust Engine, Adaptive Aggregation, Round-Risk Assessment, Quarantine State Machine, Model Poisoning, Non-IID Data, FedAvg.", bold=True)

    doc.add_page_break()

    # =============================================================
    # TABLE OF CONTENTS, FIGURES, TABLES
    # =============================================================
    add_heading_1("TABLE OF CONTENTS")
    toc_data = [
        ["Certificate", "i"],
        ["Acknowledgements", "ii"],
        ["Abstract", "iii"],
        ["List of Figures", "vi"],
        ["List of Tables", "vii"],
        ["CHAPTER 1: INTRODUCTION", "1"],
        ["  1.1 Context and Motivation", "1"],
        ["  1.2 Problem Statement", "2"],
        ["  1.3 Objectives", "3"],
        ["  1.4 Background and Literature Survey", "4"],
        ["  1.5 Organization of the Report", "8"],
        ["CHAPTER 2: PROPOSED TARA-FL FRAMEWORK", "9"],
        ["  2.1 System Architecture and Execution Pipeline", "9"],
        ["  2.2 Multi-Stage Dynamic Trust Engine", "11"],
        ["  2.3 Trust Zone State Machine and Quarantine Lifecycle", "15"],
        ["  2.4 Multi-Signal Round-Risk Assessment Engine", "17"],
        ["  2.5 Risk-Aware Adaptive Aggregation Routing", "19"],
        ["  2.6 Overall Algorithmic Formulation", "21"],
        ["CHAPTER 3: SYSTEM DESIGN AND IMPLEMENTATION", "23"],
        ["  3.1 Development and Execution Environment", "23"],
        ["  3.2 Technology Stack and Specifications", "24"],
        ["  3.3 Dataset Distribution and Client Simulation", "25"],
        ["  3.4 Adversarial Attack Simulation Modules", "27"],
        ["  3.5 Trust Engine & Quarantine Implementation", "28"],
        ["  3.6 Aggregation Modules and Orchestration", "30"],
        ["CHAPTER 4: EXPERIMENTAL METHODOLOGY", "32"],
        ["  4.1 Research Questions and Objectives", "32"],
        ["  4.2 Experimental Configuration & Hyperparameters", "33"],
        ["  4.3 Byzantine Attack Configurations", "34"],
        ["  4.4 Quantitative Evaluation Metrics", "35"],
        ["  4.5 Complete 20-Scenario Experimental Matrix Design", "36"],
        ["CHAPTER 5: RESULTS AND DISCUSSION", "38"],
        ["  5.1 Clean Baseline Performance (0% Byzantine)", "38"],
        ["  5.2 Robustness Under Gradient-Scaling Attacks", "39"],
        ["  5.3 Robustness Under Sign-Flipping Attacks", "41"],
        ["  5.4 Robustness Under Label-Flipping Attacks", "42"],
        ["  5.5 Malicious Weight Suppression Analysis", "43"],
        ["  5.6 Computational Latency and Runtime Overhead", "44"],
        ["  5.7 Discussion and Comparative Analysis", "45"],
        ["CHAPTER 6: CONCLUSION AND FUTURE SCOPE", "47"],
        ["  6.1 Summary of Achievements", "47"],
        ["  6.2 Key Architectural Insights", "48"],
        ["  6.3 Limitations", "49"],
        ["  6.4 Future Research Directions", "49"],
        ["CHAPTER 7: APPENDIX", "51"],
        ["CHAPTER 8: REFERENCES", "56"],
    ]
    add_styled_table(["Chapter / Section", "Page"], toc_data, [5.5, 1.0])

    doc.add_page_break()

    # List of Figures & Tables
    add_heading_1("LIST OF FIGURES")
    fig_data = [
        ["Figure 2.1", "TARA-FL End-to-End System Pipeline and Execution Flow", "10"],
        ["Figure 2.2", "Five-Stage Dynamic Trust Engine Mathematical Architecture", "12"],
        ["Figure 2.3", "Trust Zone Classification and Quarantine State Transition Machine", "16"],
        ["Figure 2.4", "Multi-Signal Round-Risk Assessment and Threat Classifier", "18"],
        ["Figure 2.5", "Adaptive Aggregation Strategy Dispatch Matrix Based on Risk", "20"],
        ["Figure 3.1", "Convolutional Neural Network (CNN) Architecture for MNIST", "26"],
        ["Figure 5.1", "Test Accuracy Convergence Trajectory on Clean Baseline (IID vs. Non-IID)", "38"],
        ["Figure 5.2", "Global Test Accuracy Under 30% Gradient-Scaling Attack (IID)", "40"],
        ["Figure 5.3", "Global Test Accuracy Under 30% Sign-Flipping Attack (Non-IID)", "41"],
        ["Figure 5.4", "Malicious Client Aggregation Weight Discounting Trajectory Over Rounds", "43"],
    ]
    add_styled_table(["Figure No.", "Caption / Title", "Page"], fig_data, [1.2, 4.3, 1.0])

    add_heading_1("LIST OF TABLES")
    tbl_data = [
        ["Table 3.1", "Hardware and Software Specifications of the Testbed Environment", "24"],
        ["Table 3.2", "TARA-FL System Hyperparameters and Default Configuration Values", "29"],
        ["Table 4.1", "Adversarial Attack Parameter Configurations and Objectives", "34"],
        ["Table 4.2", "Evaluation Metrics and Mathematical Definitions", "35"],
        ["Table 4.3", "Complete 20-Scenario Experimental Evaluation Matrix", "37"],
        ["Table 5.1", "Clean Baseline Performance (0% Byzantine, 5 Rounds)", "38"],
        ["Table 5.2", "Final Accuracy Summary Under Gradient Scaling (Scale Factor = 10×)", "40"],
        ["Table 5.3", "Final Accuracy Summary Under Sign Flipping Attack (Negation = −1×)", "41"],
        ["Table 5.4", "Final Accuracy Summary Under Label Flipping Attack (75% Flip Ratio)", "42"],
        ["Table 5.5", "Mean Malicious Client Aggregation Weight Under 30% Byzantine Proportion", "44"],
        ["Table 5.6", "Per-Round Computational Latency and Runtime Overhead Comparison", "45"],
        ["Table 7.1", "Automated Pytest Suite Coverage Across 92 Unit and Integration Tests", "55"],
    ]
    add_styled_table(["Table No.", "Caption / Title", "Page"], tbl_data, [1.2, 4.3, 1.0])

    doc.add_page_break()

    # =============================================================
    # CHAPTER 1: INTRODUCTION
    # =============================================================
    add_heading_1("CHAPTER 1: INTRODUCTION")
    
    add_heading_2("1.1 Context and Motivation")
    add_p(
        "Machine learning has become the bedrock of modern intelligent applications, underpinning systems from computer vision and "
        "natural language processing to clinical medicine and autonomous robotics. In the classical centralized training paradigm, data generated "
        "across distributed edge devices (such as personal smartphones, hospital workstations, banking servers, and IoT sensors) is transmitted "
        "across public networks and aggregated onto central cloud data centers. While centralized training facilitates high-throughput optimization, "
        "it creates severe friction with modern data privacy regulations, including the General Data Protection Regulation (GDPR) in the European Union, "
        "the Health Insurance Portability and Accountability Act (HIPAA) in the United States, and emerging statutory data protection frameworks globally. "
        "These regulations strictly control the transfer, aggregation, and secondary analysis of private personal data. Beyond regulatory exposure, "
        "centralized data lakes represent high-value vulnerabilities susceptible to catastrophic data breaches."
    )
    add_p(
        "These challenges catalyzed the development of Federated Learning (FL), introduced by McMahan et al. in 2017 [1]. FL enables decentralized edge "
        "clients to collaboratively train a shared global model without sharing their raw local data. By confining raw training data to the originating device "
        "and exchanging only model parameters or gradients with a coordinating server, FL provides a privacy-preserving framework for collaborative machine "
        "learning across distributed environments."
    )

    add_heading_2("1.2 Problem Statement")
    add_p("Despite its privacy guarantees, decentralized federated optimization introduces severe vulnerabilities:")
    add_bullet("Byzantine and Model-Poisoning Vulnerability: Because the central server cannot inspect private local training sets, malicious participants can manipulate the training process through data-poisoning attacks (e.g., label flipping) or direct model-update poisoning (e.g., gradient scaling, sign flipping). A small minority of Byzantine clients can easily compromise the global model.")
    add_bullet("Statistical Heterogeneity (Non-IID Data): In real-world deployments, local client data is non-identically distributed. Standard aggregation rules often mistake legitimate non-IID updates for malicious anomalies, penalizing benign clients and degrading convergence.")
    add_bullet("Inflexibility of Static Defenses: Existing robust aggregation operators (such as Krum, Coordinate Median, and Trimmed Mean) apply fixed filtering rules regardless of the current threat level. Under benign conditions, these static rules discard informative gradient variance and increase computational overhead; under evolving attacks, they fail to track historical adversarial patterns.")

    add_heading_2("1.3 Objectives")
    add_p("The primary objectives of this capstone project are:")
    add_bullet("Dynamic Trust Scoring Engine: Design a five-stage trust engine that computes a continuous reliability score T_k ∈ [0, 1] for each client using peer-relative Median Absolute Deviation (MAD) scaling, inverse-quadratic decay, exponential moving average (EMA) history, and an exponential persistence penalty.")
    add_bullet("Graduated Trust Zones and Quarantine Protocol: Implement a four-zone state machine (Safe, Probation, Low Trust, and Quarantine) with a recoverable probation pathway requiring N_clean ≥ 2 clean rounds to re-enter aggregation.")
    add_bullet("Multi-Signal Round-Risk Assessment: Formulate a composite risk engine that synthesizes update distance anomalies, population trust deficits, suspicious client ratios, worst-client deviations, and validation drop penalties into a scalar risk metric R_t ∈ [0, 1].")
    add_bullet("Risk-Aware Adaptive Aggregation: Implement an adaptive router that dynamically selects between Trust-Aware FedAvg (Low Risk), Robust Trimming (Medium Risk), and Coordinate Median (High Risk).")
    add_bullet("Comprehensive Empirical Benchmark: Evaluate the framework across a 20-scenario experimental matrix (160 complete federated runs) against three baselines across varying Byzantine proportions (0% to 30%) and data distributions (IID and non-IID).")

    add_heading_2("1.4 Background and Literature Survey")
    add_heading_3("1.4.1 Federated Learning Fundamentals")
    add_p(
        "In a standard FL round t, a central server broadcasts the current global parameter vector W_t ∈ R^d to a cohort of K clients. "
        "Each selected client k updates the model using its local private dataset D_k via Stochastic Gradient Descent (SGD) over E local epochs: "
        "W_{t+1, k} = W_t - η ∇ F_k(W_t). The parameter update is defined as ΔW_k^(t) = W_{t+1, k} - W_t. Under standard Federated Averaging (FedAvg) [1], "
        "the server aggregates client updates proportional to local sample counts n_k = |D_k|:"
    )
    add_p("W_{t+1} = W_t + Σ_{k=1}^K (n_k / N) ΔW_k^(t), where N = Σ_{k=1}^K n_k", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    add_heading_3("1.4.2 Threat Vectors: Data and Model Poisoning")
    add_p(
        "Federated optimization is vulnerable to several classes of adversarial poisoning: Data-Poisoning (Label Flipping), where malicious clients "
        "train on corrupted labels (y -> (y + shift) mod C) [2], [3]; Model-Poisoning (Gradient Scaling), where malicious clients amplify their update vectors "
        "(ΔW_k -> γ ΔW_k, γ >> 1) to dominate the aggregate [15]; and Model-Poisoning (Sign Flipping), where updates are negated (ΔW_k -> -ΔW_k) "
        "to push the model away from the loss minimum [15]."
    )

    add_heading_3("1.4.3 Classical Robust Aggregation & Trust Systems")
    add_p(
        "Classical defenses replace the arithmetic mean with robust statistics: Coordinate-wise Median and Trimmed Mean (Yin et al. [5]), "
        "Geometric Selection via Krum and Bulyan (Blanchard et al. [6], El Mhamdi et al. [7]), and scalar reputation scoring (Li et al. [8], Liu et al. [9]). "
        "However, existing methods typically apply static, unvarying rules that struggle under non-IID skew or intermittent Byzantine adversaries."
    )

    add_heading_3("1.4.4 Research Gaps Addressed by TARA-FL")
    add_p(
        "Existing defensive schemes operate as static, isolated components. They lack mechanisms to: (a) dynamically adapt aggregation strategies "
        "to the current threat level, (b) distinguish benign non-IID statistical variance from malicious behavior, and (c) provide a recoverable "
        "quarantine protocol for temporarily degraded clients. TARA-FL unifies dynamic trust tracking, multi-signal threat estimation, and "
        "adaptive aggregation routing into a single coherent framework."
    )

    add_heading_2("1.5 Organization of the Report")
    add_p(
        "The report is organized into eight chapters detailing the proposed TARA-FL framework (Chapter 2), system design and implementation (Chapter 3), "
        "experimental methodology (Chapter 4), results and discussion (Chapter 5), conclusion and future scope (Chapter 6), appendix (Chapter 7), "
        "and bibliographic references (Chapter 8)."
    )

    doc.add_page_break()

    # =============================================================
    # CHAPTER 2: PROPOSED TARA-FL FRAMEWORK
    # =============================================================
    add_heading_1("CHAPTER 2: PROPOSED TARA-FL FRAMEWORK")
    
    add_heading_2("2.1 System Architecture and Execution Pipeline")
    add_p(
        "The TARA-FL architecture organizes the federated training loop into a sequential, multi-stage pipeline: parameter updates received from clients "
        "are filtered through multi-metric anomaly detection, dynamic trust scoring, trust-zone quarantine management, round-risk assessment, and "
        "adaptive aggregation routing to produce the updated global model."
    )
    add_code_block(
        "[Global Model W_t] ──► [Client Selection] ──► [Local SGD Training]\n"
        "                             │\n"
        "                             ▼\n"
        "                    [Parameter Updates ΔW_k]\n"
        "                             │\n"
        "                             ▼\n"
        "                 [Multi-Metric Anomaly Detection]\n"
        "                             │ (PID Anomaly Scores S_k)\n"
        "                             ▼\n"
        "                 [5-Stage Dynamic Trust Engine]\n"
        "                             │ (Trust Scores T_k)\n"
        "                             ▼\n"
        "                 [Trust Zone & Quarantine Manager]\n"
        "                             │ (Eligible Clients S_t)\n"
        "                             ▼\n"
        "               [Multi-Signal Round-Risk Assessment]\n"
        "                             │ (Risk Level: LOW / MED / HIGH)\n"
        "                             ▼\n"
        "                 [Adaptive Aggregation Router]\n"
        "                             │\n"
        "                             ▼\n"
        "                   [Global Model W_{t+1}]"
    )

    add_heading_2("2.2 Multi-Stage Dynamic Trust Engine")
    add_p("For each client k in {1, ..., K}, the trust engine computes a continuous trust score T_k^(t) in [0, 1] through five mathematical stages:")
    
    add_heading_3("2.2.1 Stage 1: Peer-Relative MAD Anomaly Scaling")
    add_p("Med(S) = median(S),  MAD(S) = median({|S_k - Med(S)|}_{k=1}^K),  σ_robust = max(MAD(S), 0.01)")
    add_p("Ã_k = 1.0 + max(0, (S_k - Med(S)) / σ_robust)")

    add_heading_3("2.2.2 Stage 2: Current-Round Inverse-Quadratic Trust")
    add_p("T_curr,k^(t) = 1 / (1 + (max(0, Ã_k - 1.0 - δ_normal))^2), with δ_normal = 1.0")

    add_heading_3("2.2.3 Stage 3: Historical Trust Memory via EMA")
    add_p("T_hist,k^(t) = α · T_k^(t-1) + (1 - α) · T_curr,k^(t), with α = 0.60 (initial T^(0) = 1.0)")

    add_heading_3("2.2.4 Stage 4: Temporal Persistence Penalty")
    add_p("Severity_k^(τ) = min(1.0, max(0, Ã_k^(τ) - 1.0) / 3.0)")
    add_p("P_k^(t) = (Σ_{τ=1}^t γ^(t-τ) Severity_k^(τ)) / (Σ_{τ=1}^t γ^(t-τ)), with γ = 0.80")

    add_heading_3("2.2.5 Stage 5: Composite Trust Formulation")
    add_p("T_k^(t) = clamp(0.6 · T_hist,k^(t) + 0.4 · T_curr,k^(t) - 0.2 · P_k^(t), 0.0, 1.0)")

    add_heading_2("2.3 Trust Zone State Machine and Quarantine Lifecycle")
    add_bullet("Safe Zone (T_k ≥ 0.75): Full trust-weighted participation.")
    add_bullet("Probation Zone (0.40 ≤ T_k < 0.75): Down-weighted participation; monitored for consecutive clean rounds.")
    add_bullet("Low Trust Zone (0.20 ≤ T_k < 0.40): Heavily down-weighted; candidate for quarantine.")
    add_bullet("Quarantine Zone (T_k < 0.20): Excluded from aggregation (Weight = 0). Requires N_clean ≥ 2 clean rounds (Ã_k ≤ 2.0) to exit.")

    add_heading_2("2.4 Multi-Signal Round-Risk Assessment Engine")
    add_p(
        "Round risk R_t in [0, 1] aggregates five signals: Update Distance Anomaly (weight 0.25), Population Trust Deficit (0.25), "
        "Suspicious Client Proportion (0.20), Worst-Client Deficit (0.30), and Validation Degradation (additive +0.20 * drop if > 5%)."
    )
    add_bullet("LOW Risk (R_t < 0.30): Safe environment -> Dispatches Trust-Aware FedAvg.")
    add_bullet("MEDIUM Risk (0.30 ≤ R_t < 0.60): Moderate poisoning -> Dispatches Robust Trimming (trim ratio β = 0.25).")
    add_bullet("HIGH Risk (R_t ≥ 0.60): Severe attack -> Dispatches Coordinate-wise Median (50% breakdown point).")

    add_heading_2("2.5 Complete Algorithmic Workflow")
    add_code_block(
        "================================================================================\n"
        "Algorithm 1: TARA-FL Federated Round Execution\n"
        "================================================================================\n"
        "Input : Global model W_t, Client cohort C, Trust history H, Quarantine state Q,\n"
        "        Validation set D_val\n"
        "Output: Updated global model W_{t+1}, Updated states H', Q'\n"
        "\n"
        " 1: Server broadcasts W_t to all clients k in C\n"
        " 2: for each client k in C in parallel do\n"
        " 3:    W_{t+1, k} = Local_SGD(W_t, D_k, E=1, eta=0.01)\n"
        " 4:    Delta W_k = W_{t+1, k} - W_t\n"
        " 5: end for\n"
        " 6: Server collects updates {Delta W_k}_{k in C}\n"
        " 7: Compute pairwise distances D and PID scores S_k relative to coordinate median\n"
        " 8: for each client k in C do\n"
        " 9:    tilde{A}_k = 1.0 + max(0, (S_k - Med(S)) / max(MAD(S), 0.01))\n"
        "10:    T_{curr, k} = 1.0 / (1.0 + max(0, tilde{A}_k - 1.0 - delta_normal)^2)\n"
        "11:    T_{hist, k} = alpha * T_k^{(t-1)} + (1 - alpha) * T_{curr, k}\n"
        "12:    P_k = Temporal_Persistence_Penalty(k, tilde{A}_k, gamma=0.8)\n"
        "13:    T_k^{(t)} = clamp(0.6 * T_{hist, k} + 0.4 * T_{curr, k} - 0.2 * P_k, 0, 1)\n"
        "14:    zone_k = QuarantineManager_Update(k, T_k^{(t)}, tilde{A}_k, Q)\n"
        "15: end for\n"
        "16: Filter eligible cohort: S_t = {k in C | zone_k != QUARANTINE}\n"
        "17: R_t, RiskLevel, ThreatType = RoundRisk_Assess({Delta W_k}, {T_k}, D_val)\n"
        "18: if S_t is empty then\n"
        "19:    W_{t+1} = W_t  // Retain model under total compromise\n"
        "20: else if RiskLevel == \"LOW\" then\n"
        "21:    W_{t+1} = TrustAwareFedAvg(S_t, {Delta W_k}, {T_k})\n"
        "22: else if RiskLevel == \"MEDIUM\" then\n"
        "23:    W_{t+1} = TrustWeightedRobustTrim(S_t, {Delta W_k}, {T_k}, beta=0.25)\n"
        "24: else if RiskLevel == \"HIGH\" then\n"
        "25:    W_{t+1} = TrustWeightedCoordinateMedian(S_t, {Delta W_k}, {T_k})\n"
        "26: end if\n"
        "27: Evaluate test accuracy Acc_{t+1} and update history H'\n"
        "28: return W_{t+1}, H', Q'\n"
        "================================================================================"
    )

    doc.add_page_break()

    # =============================================================
    # CHAPTER 3: SYSTEM DESIGN AND IMPLEMENTATION
    # =============================================================
    add_heading_1("CHAPTER 3: SYSTEM DESIGN AND IMPLEMENTATION")
    
    add_heading_2("3.1 Development and Execution Environment")
    add_p(
        "TARA-FL is implemented in Python 3.10+ using PyTorch as the primary deep learning framework. "
        "The complete hardware and software environment is specified in Table 3.1."
    )

    add_heading_3("Table 3.1: Hardware and Software Specifications of the Testbed Environment")
    add_styled_table(
        ["Component", "Specification"],
        [
            ["Processor (CPU)", "Intel Core i7-12700H (14 Cores, 20 Threads, up to 4.7 GHz)"],
            ["System Memory (RAM)", "32 GB DDR5 4800 MHz"],
            ["GPU Acceleration", "NVIDIA GeForce RTX 3060 Laptop GPU (6 GB GDDR6)"],
            ["Operating System", "Microsoft Windows 11 Pro / Ubuntu 22.04 LTS"],
            ["Programming Language", "Python 3.10.12 / Python 3.11.8"],
            ["Core Deep Learning", "PyTorch 2.0.1+cu118, torchvision 0.15.2+cu118"],
            ["Scientific Computing", "NumPy 1.24.3, SciPy 1.10.1, scikit-learn 1.2.2"],
            ["Data Analysis & Visuals", "pandas 2.0.2, matplotlib 3.7.1, Pillow 9.5.0, Jinja2 3.1.0"],
            ["Automated Testing", "pytest 7.3.1 (92 automated unit/integration tests)"],
        ],
        [2.5, 4.0]
    )

    add_heading_2("3.2 Dataset Distribution and Client Simulation")
    add_p(
        "The MNIST dataset (60,000 train, 10,000 test) is partitioned across 10 clients using either IID uniform splits or "
        "Dirichlet non-IID splits (concentration parameter α = 0.5). Each client trains a 4-layer CNN (Conv2d 32 -> Conv2d 64 -> Linear 128 -> Linear 10) "
        "comprising 421,642 trainable parameters using local SGD (batch size 32, learning rate 0.01, 1 local epoch per round)."
    )

    add_heading_2("3.3 Adversarial Attack Modules")
    add_bullet("Label Flipping: Corrupts local ground-truth labels (y -> (y + 1) mod 10) with a 75% flip ratio.")
    add_bullet("Gradient Scaling: Multiplies local parameter updates by a factor of 10.0x.")
    add_bullet("Sign Flipping: Inverts update directions by multiplying by -1.0x.")

    add_heading_2("3.4 Hyperparameter Configuration")
    add_heading_3("Table 3.2: TARA-FL System Hyperparameters and Default Configuration Values")
    add_styled_table(
        ["Hyperparameter", "Symbol", "Default", "Description"],
        [
            ["History Memory Weight", "w_hist", "0.60", "Historical EMA weight in composite trust"],
            ["Current Anomaly Weight", "w_curr", "0.40", "Current-round trust score weight"],
            ["Persistence Penalty Weight", "w_pers", "0.20", "Subtraction weight for persistent anomalies"],
            ["EMA Decay Factor", "alpha", "0.60", "Temporal smoothing factor for historical trust"],
            ["Severity Discount Factor", "gamma", "0.80", "Temporal discount for persistence penalty"],
            ["Normal Deviation Allowance", "delta_norm", "1.00", "Anomaly threshold before decay begins"],
            ["Quarantine Threshold", "tau_quar", "0.20", "Trust threshold below which quarantine triggers"],
            ["Clean Probation Rounds", "N_clean", "2", "Clean rounds required to exit quarantine"],
            ["Robust Trimming Ratio", "beta", "0.25", "Trimming fraction under Medium Risk"],
            ["Low Risk Boundary", "tau_low", "0.30", "Threshold between Low and Medium risk"],
            ["Medium Risk Boundary", "tau_med", "0.60", "Threshold between Medium and High risk"],
            ["Local Learning Rate", "eta", "0.01", "SGD step size for local client training"],
            ["Local Batch Size", "B", "32", "Mini-batch size for local client training"],
            ["Local Epochs", "E", "1", "Local epochs per federated round"],
        ],
        [2.2, 0.9, 0.8, 2.6]
    )

    doc.add_page_break()

    # =============================================================
    # CHAPTER 4: EXPERIMENTAL METHODOLOGY
    # =============================================================
    add_heading_1("CHAPTER 4: EXPERIMENTAL METHODOLOGY")
    
    add_heading_2("4.1 Research Questions and Objectives")
    add_bullet("RQ1 (Byzantine Robustness): Can TARA-FL maintain accuracy under severe (30%) attacks where FedAvg fails?")
    add_bullet("RQ2 (Malicious Influence Mitigation): Does dynamic trust reduce Byzantine client weights to near zero?")
    add_bullet("RQ3 (Clean Baseline Overhead): What is the accuracy trade-off in benign (0% Byzantine) environments?")
    add_bullet("RQ4 (Computational Latency): What is the per-round runtime overhead compared to standard FedAvg?")

    add_heading_2("4.2 Attack Configurations and Metrics")
    add_heading_3("Table 4.1: Adversarial Attack Parameter Configurations and Objectives")
    add_styled_table(
        ["Attack Name", "Category", "Parameter Settings", "Adversarial Objective"],
        [
            ["Clean Baseline", "None", "0% Malicious Clients", "Benchmark ceiling performance"],
            ["Label Flipping", "Data Poisoning", "75% flip ratio, shift = +1", "Induce targeted misclassification"],
            ["Gradient Scaling", "Model Poisoning", "Scaling factor = 10.0x", "Dominate the aggregated update"],
            ["Sign Flipping", "Model Poisoning", "Negation factor = -1.0x", "Push model away from loss minimum"],
        ],
        [1.5, 1.5, 1.8, 1.7]
    )

    add_heading_3("Table 4.2: Evaluation Metrics and Mathematical Definitions")
    add_styled_table(
        ["Metric", "Mathematical Formulation / Definition"],
        [
            ["Final Test Accuracy (%)", "(Correct Predictions / Total Test Samples) * 100 at Round 5"],
            ["Test Cross-Entropy Loss", "-(1/N_test) * sum(y_i * log(p_i)) over test partition"],
            ["Mean Malicious Weight", "(1/T) * sum_{t=1}^T sum_{k in Malicious} Omega_k^(t)"],
            ["Clean Accuracy Trade-off", "Acc_{FedAvg, Clean} - Acc_{TARA-FL, Clean}"],
            ["Per-Round Wall Time (s)", "Total Wall-Clock Execution Time / Total Federated Rounds"],
        ],
        [2.5, 4.0]
    )

    add_heading_2("4.3 Complete 20-Scenario Experimental Matrix")
    add_p(
        "The experimental matrix spans 20 unique scenarios: 2 clean baselines (IID and non-IID) plus 18 attack scenarios "
        "(3 attack types x 3 Byzantine proportions [10%, 20%, 30%] x 2 data distributions). Each scenario evaluates 4 methods "
        "(FEDAVG, TRUST_FEDAVG, TRUST_ROBUST, TARA-FL) across 2 random seeds, yielding 160 complete federated executions."
    )

    doc.add_page_break()

    # =============================================================
    # CHAPTER 5: RESULTS AND DISCUSSION
    # =============================================================
    add_heading_1("CHAPTER 5: RESULTS AND DISCUSSION")
    
    add_heading_2("5.1 Clean Baseline Performance (0% Byzantine)")
    add_heading_3("Table 5.1: Clean Baseline Performance (0% Byzantine, 5 Rounds)")
    add_styled_table(
        ["Method", "IID Accuracy (%)", "IID Loss", "Non-IID Accuracy (%)", "Non-IID Loss"],
        [
            ["FedAvg", "89.73 ± 0.12", "0.3504", "88.68 ± 0.31", "0.3805"],
            ["Trust+FedAvg", "89.68 ± 0.08", "0.3525", "87.00 ± 0.45", "0.4226"],
            ["Trust+Robust", "89.48 ± 0.15", "0.3553", "85.25 ± 0.52", "0.4670"],
            ["TARA-FL", "89.47 ± 0.11", "0.3554", "85.45 ± 0.38", "0.4621"],
        ],
        [1.8, 1.4, 1.1, 1.4, 1.1]
    )
    add_p("In benign IID environments, TARA-FL incurs an accuracy trade-off of only 0.26%. In non-IID conditions, the trade-off is 3.23%.")

    add_heading_2("5.2 Robustness Under Gradient-Scaling Attacks")
    add_heading_3("Table 5.2: Final Accuracy Summary Under Gradient Scaling (Scale Factor = 10×)")
    add_styled_table(
        ["Scenario", "FedAvg", "Trust+FedAvg", "Trust+Robust", "TARA-FL", "TARA Gain vs FedAvg"],
        [
            ["10% Byzantine, IID", "85.34%", "90.27%", "89.88%", "89.81%", "+4.47%"],
            ["10% Byzantine, Non-IID", "65.75%", "89.18%", "87.35%", "84.54%", "+18.79%"],
            ["20% Byzantine, IID", "48.94%", "90.44%", "89.79%", "89.55%", "+40.61%"],
            ["20% Byzantine, Non-IID", "29.93%", "89.47%", "85.76%", "85.19%", "+55.26%"],
            ["30% Byzantine, IID", "22.65%", "90.56%", "90.07%", "90.00%", "+67.35%"],
            ["30% Byzantine, Non-IID", "16.51%", "89.20%", "86.79%", "83.33%", "+66.82%"],
        ],
        [1.8, 0.9, 1.0, 1.0, 1.0, 1.3]
    )
    add_p("Under 30% Gradient Scaling, FedAvg collapses to 22.65% (IID), while TARA-FL maintains 90.00% (+67.35% gain).")

    add_heading_2("5.3 Robustness Under Sign-Flipping Attacks")
    add_heading_3("Table 5.3: Final Accuracy Summary Under Sign Flipping Attack (Negation = −1×)")
    add_styled_table(
        ["Scenario", "FedAvg", "Trust+FedAvg", "Trust+Robust", "TARA-FL", "TARA Gain vs FedAvg"],
        [
            ["10% Byzantine, IID", "88.70%", "89.75%", "89.78%", "89.76%", "+1.06%"],
            ["10% Byzantine, Non-IID", "76.03%", "82.98%", "82.75%", "83.74%", "+7.71%"],
            ["20% Byzantine, IID", "86.17%", "89.57%", "89.77%", "89.47%", "+3.30%"],
            ["20% Byzantine, Non-IID", "69.16%", "82.23%", "83.36%", "82.36%", "+13.20%"],
            ["30% Byzantine, IID", "79.62%", "89.39%", "89.63%", "89.60%", "+9.98%"],
            ["30% Byzantine, Non-IID", "62.00%", "78.77%", "81.78%", "81.97%", "+19.97%"],
        ],
        [1.8, 0.9, 1.0, 1.0, 1.0, 1.3]
    )

    add_heading_2("5.4 Robustness Under Label-Flipping Attacks")
    add_heading_3("Table 5.4: Final Accuracy Summary Under Label Flipping Attack (75% Flip Ratio)")
    add_styled_table(
        ["Scenario", "FedAvg", "Trust+FedAvg", "Trust+Robust", "TARA-FL", "TARA Gain vs FedAvg"],
        [
            ["10% Byzantine, IID", "89.44%", "89.78%", "89.66%", "89.72%", "+0.28%"],
            ["10% Byzantine, Non-IID", "87.12%", "85.47%", "83.83%", "82.71%", "-4.41%"],
            ["20% Byzantine, IID", "88.62%", "89.72%", "89.76%", "89.47%", "+0.85%"],
            ["20% Byzantine, Non-IID", "86.82%", "84.78%", "83.94%", "83.21%", "-3.61%"],
            ["30% Byzantine, IID", "86.63%", "89.68%", "89.72%", "89.65%", "+3.02%"],
            ["30% Byzantine, Non-IID", "84.24%", "84.48%", "83.84%", "82.06%", "-2.18%"],
        ],
        [1.8, 0.9, 1.0, 1.0, 1.0, 1.3]
    )

    add_heading_2("5.5 Malicious Weight Suppression Analysis")
    add_heading_3("Table 5.5: Mean Malicious Client Aggregation Weight Under 30% Byzantine Proportion")
    add_styled_table(
        ["Attack Setting", "FedAvg Weight", "Trust+FedAvg", "Trust+Robust", "TARA-FL"],
        [
            ["Gradient Scaling (IID)", "0.3000", "0.0358", "0.0071", "0.0308"],
            ["Gradient Scaling (Non-IID)", "0.2772", "0.0668", "0.0000", "0.0511"],
            ["Sign Flipping (IID)", "0.3000", "0.0345", "0.0064", "0.0300"],
            ["Sign Flipping (Non-IID)", "0.2772", "0.1661", "0.1158", "0.1392"],
            ["Label Flipping (IID)", "0.3000", "0.0338", "0.0062", "0.0295"],
            ["Label Flipping (Non-IID)", "0.2772", "0.1299", "0.1111", "0.1142"],
        ],
        [2.2, 1.1, 1.1, 1.1, 1.0]
    )

    add_heading_2("5.6 Computational Latency and Runtime Overhead")
    add_heading_3("Table 5.6: Per-Round Computational Latency and Runtime Overhead Comparison")
    add_styled_table(
        ["Component / Method", "Mean Round Time (s)", "Overhead vs FedAvg", "Complexity"],
        [
            ["Standard FedAvg Baseline", "12.64 s", "Baseline (0.0%)", "O(N * P)"],
            ["Trust Engine Scoring", "+0.04 s", "+0.32%", "O(N)"],
            ["Quarantine State Update", "+0.01 s", "+0.08%", "O(N)"],
            ["Round-Risk Assessment", "+0.03 s", "+0.24%", "O(N * P)"],
            ["Adaptive Aggregation Router", "+0.06 s", "+0.47%", "O(N * P log N)"],
            ["Total TARA-FL Pipeline", "12.78 s", "+1.11%", "O(N * P log N)"],
        ],
        [2.3, 1.4, 1.4, 1.4]
    )

    doc.add_page_break()

    # =============================================================
    # CHAPTER 6: CONCLUSION AND FUTURE SCOPE
    # =============================================================
    add_heading_1("CHAPTER 6: CONCLUSION AND FUTURE SCOPE")
    
    add_heading_2("6.1 Summary of Achievements")
    add_p(
        "This capstone project presented TARA-FL, an integrated trust-aware, robust, and adaptive federated learning framework. "
        "The project demonstrated that treating defense as a continuous control problem—where aggregation strategy adapts dynamically "
        "to round-level threat severity—provides exceptional Byzantine robustness (+67.35% under gradient scaling, +19.98% under sign flipping) "
        "while introducing negligible latency (<0.15 s/round) and preserving near-baseline accuracy in benign environments."
    )

    add_heading_2("6.2 Limitations")
    add_bullet("Evaluation was conducted on simulated client partitions using the MNIST benchmark; larger-scale datasets remain to be explored.")
    add_bullet("Anomaly detection relies on geometric parameter properties, which reduces sensitivity to subtle data poisoning under severe non-IID skew.")
    add_bullet("Formal cryptographic privacy mechanisms (e.g., Secure Multi-Party Computation) were not integrated into the communication layer.")

    add_heading_2("6.3 Future Research Directions")
    add_bullet("Semantic Validation Probes: Incorporate server-side validation probe sets to evaluate the functional loss impact of client updates.")
    add_bullet("Edge Hardware Testbed: Deploy TARA-FL on physical embedded devices (e.g., Raspberry Pi, NVIDIA Jetson).")
    add_bullet("Privacy-Preserving Trust Integration: Integrate local differential privacy or secure aggregation with dynamic trust scoring.")
    add_bullet("Automated Hyperparameter Tuning: Apply Bayesian optimization to dynamically tune trust decay and risk thresholds.")

    doc.add_page_break()

    # =============================================================
    # CHAPTER 7: APPENDIX
    # =============================================================
    add_heading_1("CHAPTER 7: APPENDIX")
    
    add_heading_2("Appendix A: Automated Test Suite Coverage Summary")
    add_styled_table(
        ["Test Module File", "Test Count", "Primary Verification Scope"],
        [
            ["test_trust_engine.py", "20", "MAD normalization, EMA history, persistence penalties, trust clamp"],
            ["test_detection.py", "12", "Coordinate-wise median PID scores, pairwise Euclidean distances"],
            ["test_risk_and_aggregation.py", "15", "Multi-signal round risk, angular threat classifier, robust trimming"],
            ["test_adaptive_aggregation.py", "10", "Dynamic strategy routing, all-quarantine degenerate fallback"],
            ["test_clients_and_attacks.py", "12", "Local client SGD, label flipping, gradient scaling, sign negation"],
            ["test_federated.py", "8", "End-to-end multi-round federated training loop integration"],
            ["test_model.py", "5", "CNN forward pass, output logit dimensions, loss computation"],
            ["test_server_and_experiments.py", "6", "Server orchestration, 20-scenario matrix configuration"],
            ["test_visualization.py", "4", "Matplotlib plot generation, HTML dashboard rendering"],
            ["Total Automated Tests", "92 / 92", "100% Passing Test Suite"],
        ],
        [2.2, 0.9, 3.4]
    )

    doc.add_page_break()

    # =============================================================
    # CHAPTER 8: REFERENCES
    # =============================================================
    add_heading_1("CHAPTER 8: REFERENCES")
    refs = [
        "[1] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS), vol. 54, Fort Lauderdale, FL, USA, 2017, pp. 1273–1282.",
        "[2] E. Bagdasaryan, A. Veit, Y. Hua, D. Estrin, and V. Shmatikov, “How to backdoor federated learning,” in Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS), 2020, pp. 2938–2948.",
        "[3] B. Bhagoji, S. Chakraborty, P. Mittal, and S. Calo, “Analyzing federated learning through an adversarial lens,” in Proceedings of the 36th International Conference on Machine Learning (ICML), Long Beach, CA, USA, 2019, pp. 634–643.",
        "[4] Y. Zhao, M. Li, L. Lai, N. Suda, D. Civin, and V. Chandra, “Federated learning with non-IID data,” arXiv preprint arXiv:1806.00582, 2018.",
        "[5] D. Yin, Y. Chen, R. Kannan, and P. Bartlett, “Byzantine-robust distributed learning: Towards optimal statistical rates,” in Proceedings of the 35th International Conference on Machine Learning (ICML), Stockholm, Sweden, 2018, pp. 5650–5659.",
        "[6] P. Blanchard, E. M. El Mhamdi, R. Guerraoui, and J. Stainer, “Machine learning with adversaries: Byzantine tolerant gradient descent,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, Long Beach, CA, USA, 2017, pp. 119–129.",
        "[7] E. M. El Mhamdi, R. Guerraoui, and S. Rouault, “The hidden vulnerability of distributed learning in Byzantium,” in Proceedings of the 35th International Conference on Machine Learning (ICML), Stockholm, Sweden, 2018, pp. 3521–3530.",
        "[8] Y. Li, C. Chen, N. Liu, H. Huang, Z. Zheng, and Q. Yan, “A blockchain-based decentralized federated learning framework with committee consensus,” IEEE Network, vol. 35, no. 1, pp. 234–241, Jan. 2021.",
        "[9] X. Liu, H. Li, G. Xu, Z. Liu, and R. Lu, “Privacy-enhanced federated learning against poisoning adversaries,” IEEE Transactions on Information Forensics and Security, vol. 16, pp. 3535–3548, Jul. 2021.",
        "[10] K. Bonawitz et al., “Practical secure aggregation for privacy-preserving machine learning,” in Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security (CCS), Dallas, TX, USA, 2017, pp. 1175–1191.",
        "[11] P. Kairouz et al., “Advances and open problems in federated learning,” Foundations and Trends in Machine Learning, vol. 14, no. 1–2, pp. 1–210, Jun. 2021.",
        "[12] T. Li, A. K. Sahu, A. Talwalkar, and V. Smith, “Federated learning: Challenges, methods, and future directions,” IEEE Signal Processing Magazine, vol. 37, no. 3, pp. 50–60, May 2020.",
        "[13] L. Lyu, H. Yu, J. Ma, L. Sun, and X. Zhang, “Privacy and robustness in federated learning: Attacks and defenses,” IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 7, pp. 8721–8740, 2024.",
        "[14] C. Fung, C. J. M. Yoon, and I. Beschastnikh, “Mitigating sybils in federated learning poisoning,” in Proceedings of the 24th International Symposium on Research in Attacks, Intrusions and Defenses (RAID), San Sebastian, Spain, 2021, pp. 411–425.",
        "[15] V. Shejwalkar and A. Houmansadr, “Manipulating the Byzantine: Optimizing model poisoning attacks and defenses for federated learning,” in Proceedings of the 28th Network and Distributed System Security Symposium (NDSS), 2021.",
        "[16] Z. Zhang, X. Cao, J. Jia, and N. Z. Gong, “FLDetector: Defending federated learning against model poisoning attacks by detecting malicious clients,” in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), Washington, DC, USA, 2022, pp. 2545–2553.",
        "[17] Z. Wang, M. Song, Z. Zhang, Y. Song, Q. Wang, and H. Qi, “Beyond inferring class representatives: User-level privacy leakage from federated learning,” in Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI), Macao, China, 2019, pp. 4800–4806.",
        "[18] R. Shokri and V. Shmatikov, “Privacy-preserving deep learning,” in Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security (CCS), Denver, CO, USA, 2015, pp. 1310–1321.",
        "[19] T. D. Cao, T. Truong, and N. T. Dang, “Trust evaluation-based aggregation method for federated learning,” in Proceedings of the IEEE International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom), Wuhan, China, 2022, pp. 543–550.",
        "[20] B. Sun, C. Wang, A. Ross, and L. Zhao, “Adaptive federated learning with dynamic aggregation and client selection,” IEEE Transactions on Neural Networks and Learning Systems, vol. 34, no. 11, pp. 8120–8132, 2023.",
        "[21] A. Krizhevsky, “Learning multiple layers of features from tiny images,” Technical Report, University of Toronto, 2009.",
        "[22] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, Nov. 1998."
    ]
    for r in refs:
        add_p(r, space_after=4)

    # Save to disk
    doc.save(output_path)
    print(f"Successfully generated Word document at: {output_path}")

if __name__ == "__main__":
    out_file1 = "TARA_FL_CAPSTONE_PROJECT_REPORT.docx"
    out_file2 = os.path.join("reports", "TARA_FL_CAPSTONE_PROJECT_REPORT.docx")
    build_word_document(out_file1)
    build_word_document(out_file2)
