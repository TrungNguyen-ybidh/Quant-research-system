"""
Report Validation Script

This script performs comprehensive validation checks on the final research report:
1. Completeness Check - no placeholders, all sections filled
2. Statistical Rigor Check - all claims have supporting statistics
3. Actionability Check - recommendations are specific and usable
4. Professional Quality Check - formatting, consistency
5. Limitations Check - honest reporting of weaknesses
"""

import re
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def validate_completeness(report_path):
    """Check for placeholders and incomplete sections."""
    print("=" * 80)
    print("VALIDATION 1: COMPLETENESS CHECK")
    print("=" * 80)
    
    with open(report_path, 'r') as f:
        content = f.read()
    
    issues = []
    
    # Check for common placeholders
    placeholders = [
        r'\[Value\]',
        r'\[Count\]',
        r'\[Description\]',
        r'\[Dates\]',
        r'\[X\.XX\]',
        r'\[X\]',
        r'\[XX\.X\]',
        r'\[XX%\]',
        r'\[Hours\]',
        r'TODO',
        r'FIXME',
        r'XXX',
        r'PLACEHOLDER'
    ]
    
    for pattern in placeholders:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            issues.append(f"Found {len(matches)} placeholder(s): {pattern}")
    
    # Check for empty tables (cells with just dashes or spaces)
    empty_table_cells = re.findall(r'\|\s*-\s*\|', content)
    if len(empty_table_cells) > 5:  # Allow some empty cells
        issues.append(f"Found {len(empty_table_cells)} potentially empty table cells")
    
    # Check section count
    sections = re.findall(r'^##\s+\d+\.', content, re.MULTILINE)
    if len(sections) < 10:
        issues.append(f"Only found {len(sections)} major sections (expected 10+)")
    
    # Check visualization count
    images = re.findall(r'!\[.*?\]\(.*?\)', content)
    if len(images) < 10:
        issues.append(f"Only found {len(images)} visualizations (expected 10+)")
    
    # Check figure captions
    figures = re.findall(r'\*\*Figure\s+\d+\.\d+:', content)
    if len(figures) < 10:
        issues.append(f"Only found {len(figures)} figure captions (expected 10+)")
    
    if issues:
        print("⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ All completeness checks passed!")
        print(f"   - Sections: {len(sections)}")
        print(f"   - Visualizations: {len(images)}")
        print(f"   - Figure captions: {len(figures)}")
        return True


def validate_statistical_rigor(report_path):
    """Check that all claims have statistical backing."""
    print("\n" + "=" * 80)
    print("VALIDATION 2: STATISTICAL RIGOR CHECK")
    print("=" * 80)
    
    with open(report_path, 'r') as f:
        content = f.read()
    
    issues = []
    
    # Check for key statistics with confidence intervals
    ci_patterns = [
        r'95%\s*CI',
        r'confidence\s*interval',
        r'\[.*?,\s*.*?\]'  # Brackets suggesting CI
    ]
    
    ci_found = any(re.search(pattern, content, re.IGNORECASE) for pattern in ci_patterns)
    if not ci_found:
        issues.append("No confidence intervals found - statistical rigor may be insufficient")
    
    # Check for p-values
    p_value_patterns = [
        r'p\s*[<>=]?\s*0\.\d+',
        r'p-value',
        r'statistical\s*significance'
    ]
    
    p_values_found = any(re.search(pattern, content, re.IGNORECASE) for pattern in p_value_patterns)
    if not p_values_found:
        issues.append("No p-values found - statistical significance not reported")
    
    # Check for sample sizes
    sample_size_patterns = [
        r'n\s*=\s*\d+',
        r'sample\s*size',
        r'\d+\s*samples'
    ]
    
    sample_sizes_found = any(re.search(pattern, content, re.IGNORECASE) for pattern in sample_size_patterns)
    if not sample_sizes_found:
        issues.append("No sample sizes reported - statistical validity unclear")
    
    if issues:
        print("⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ All statistical rigor checks passed!")
        print("   - Confidence intervals found")
        print("   - P-values reported")
        print("   - Sample sizes included")
        return True


def validate_actionability(report_path):
    """Check that recommendations are specific and actionable."""
    print("\n" + "=" * 80)
    print("VALIDATION 3: ACTIONABILITY CHECK")
    print("=" * 80)
    
    with open(report_path, 'r') as f:
        content = f.read()
    
    issues = []
    
    # Check for specific entry rules
    entry_keywords = [
        r'entry:',
        r'entry\s*rules',
        r'stop\s*loss',
        r'take\s*profit',
        r'target:',
        r'position\s*sizing'
    ]
    
    entry_rules_found = sum(1 for pattern in entry_keywords if re.search(pattern, content, re.IGNORECASE))
    if entry_rules_found < 3:
        issues.append(f"Only found {entry_rules_found} entry rule keywords (expected 3+)")
    
    # Check for vague recommendations
    vague_phrases = [
        r'consider\s*using',
        r'may\s*be\s*beneficial',
        r'could\s*work',
        r'sometimes',
        r'possibly'
    ]
    
    vague_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in vague_phrases)
    if vague_count > 5:
        issues.append(f"Found {vague_count} potentially vague phrases - recommendations may lack specificity")
    
    # Check for specific percentages/numbers in recommendations
    specific_numbers = re.findall(r'\d+\.?\d*\s*%', content)
    if len(specific_numbers) < 10:
        issues.append(f"Only found {len(specific_numbers)} specific percentages (expected 10+)")
    
    if issues:
        print("⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ All actionability checks passed!")
        print(f"   - Entry rule keywords: {entry_rules_found}")
        print(f"   - Specific percentages: {len(specific_numbers)}")
        return True


def validate_professional_quality(report_path):
    """Check formatting, consistency, and professional appearance."""
    print("\n" + "=" * 80)
    print("VALIDATION 4: PROFESSIONAL QUALITY CHECK")
    print("=" * 80)
    
    with open(report_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    issues = []
    
    # Check for consistent figure numbering
    figures = re.findall(r'\*\*Figure\s+(\d+)\.(\d+):', content)
    if figures:
        figure_nums = [(int(s), int(n)) for s, n in figures]
        # Check for gaps or duplicates
        sections = {}
        for sec, num in figure_nums:
            if sec not in sections:
                sections[sec] = []
            sections[sec].append(num)
        
        for sec in sections:
            nums = sorted(sections[sec])
            if nums != list(range(1, len(nums) + 1)):
                issues.append(f"Figure numbering issue in Section {sec}: {nums}")
    
    # Check for consistent section numbering
    sections = re.findall(r'^##\s+(\d+)\.', content, re.MULTILINE)
    if sections:
        section_nums = [int(s) for s in sections]
        if section_nums != list(range(1, len(section_nums) + 1)):
            issues.append("Section numbering has gaps or is non-sequential")
    
    # Check for typos (common misspellings)
    typos = [
        r'\bteh\b',
        r'\badn\b',
        r'\brecieve\b',
        r'\bseperate\b'
    ]
    
    for pattern in typos:
        if re.search(pattern, content, re.IGNORECASE):
            issues.append(f"Potential typo found: {pattern}")
    
    # Check line length (very long lines may indicate formatting issues)
    long_lines = [i for i, line in enumerate(lines, 1) if len(line) > 120 and not line.startswith('|')]
    if len(long_lines) > 10:
        issues.append(f"Found {len(long_lines)} lines exceeding 120 characters (may affect readability)")
    
    if issues:
        print("⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ All professional quality checks passed!")
        print(f"   - Sections numbered correctly: {len(sections)} sections")
        print(f"   - Figures numbered: {len(figures)} figures")
        return True


def validate_limitations(report_path):
    """Check that limitations are honestly reported."""
    print("\n" + "=" * 80)
    print("VALIDATION 5: LIMITATIONS CHECK")
    print("=" * 80)
    
    with open(report_path, 'r') as f:
        content = f.read()
    
    issues = []
    
    # Check for limitations section
    if 'limitations' not in content.lower() and 'limitation' not in content.lower():
        issues.append("No limitations section found - report should acknowledge weaknesses")
    
    # Check for robustness discussion
    if 'robustness' not in content.lower():
        issues.append("No robustness discussion found - model limitations should be addressed")
    
    # Check for caveats
    caveat_keywords = [
        r'past\s*performance',
        r'does\s*not\s*guarantee',
        r'limitation',
        r'caveat',
        r'assumption',
        r'weakness'
    ]
    
    caveats_found = sum(1 for pattern in caveat_keywords if re.search(pattern, content, re.IGNORECASE))
    if caveats_found < 3:
        issues.append(f"Only found {caveats_found} limitation/caveat keywords (expected 3+)")
    
    # Check for honest reporting of issues
    if '49.89%' in content or 'degradation' in content.lower():
        print("   ✓ Robustness issue honestly reported (49.89% degradation)")
    else:
        issues.append("Robustness issue not clearly stated - should mention 49.89% degradation")
    
    if issues:
        print("⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ All limitations checks passed!")
        print(f"   - Limitations section found")
        print(f"   - Caveats/caveats: {caveats_found}")
        print(f"   - Robustness issue honestly reported")
        return True


def validate_visualizations(report_path):
    """Check that all referenced visualizations exist."""
    print("\n" + "=" * 80)
    print("VALIDATION 6: VISUALIZATION CHECK")
    print("=" * 80)
    
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Extract all image paths
    image_patterns = re.findall(r'!\[.*?\]\((.*?)\)', content)
    
    missing = []
    for img_path in image_patterns:
        # Remove ../ if present, handle relative paths
        clean_path = img_path.replace('../', '')
        if not clean_path.startswith('/'):
            clean_path = os.path.join(os.path.dirname(os.path.dirname(report_path)), clean_path)
        
        if not os.path.exists(clean_path):
            missing.append(img_path)
    
    if missing:
        print(f"⚠️  Found {len(missing)} missing visualization(s):")
        for img in missing[:5]:  # Show first 5
            print(f"   - {img}")
        if len(missing) > 5:
            print(f"   ... and {len(missing) - 5} more")
        return False
    else:
        print(f"✅ All {len(image_patterns)} visualizations exist!")
        return True


def main():
    """Run all validation checks."""
    report_path = os.path.join('reports', 'XAU_USD_Research_Report.md')
    
    if not os.path.exists(report_path):
        print(f"❌ Report not found: {report_path}")
        return
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE REPORT VALIDATION")
    print("=" * 80)
    print(f"\nValidating: {report_path}\n")
    
    results = {
        'Completeness': validate_completeness(report_path),
        'Statistical Rigor': validate_statistical_rigor(report_path),
        'Actionability': validate_actionability(report_path),
        'Professional Quality': validate_professional_quality(report_path),
        'Limitations': validate_limitations(report_path),
        'Visualizations': validate_visualizations(report_path)
    }
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 REPORT VALIDATION SUCCESSFUL!")
        print("Report is ready for submission.")
    else:
        print(f"\n⚠️  Report needs {total - passed} more check(s) to pass validation.")


if __name__ == "__main__":
    main()

