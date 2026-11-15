from statement_similarity_analysis import extract_key_points, compare_key_points

def generate_difference_report(suspect_statement, witness_statement):
    suspect_points = extract_key_points(suspect_statement)
    witness_points = extract_key_points(witness_statement)
    comparison = compare_key_points(suspect_points, witness_points)

    report_lines = []
    report_lines.append("🧾 STATEMENTS REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Suspect: {suspect_statement}")
    report_lines.append(f"Witness: {witness_statement}")
    report_lines.append("-" * 60)

    for category, data in comparison.items():
        if data["matching"] or data["different"]:
            report_lines.append(f"\n CATEGORY: {category.upper()}")
            if data["matching"]:
                report_lines.append(f"   Matching elements: {', '.join(data['matching'])}")
            if data["suspect_only"]:
                report_lines.append(f"   Only in suspect:   {', '.join(data['suspect_only'])}")
            if data["witness_only"]:
                report_lines.append(f"   Only in witness:   {', '.join(data['witness_only'])}")

    return "\n".join(report_lines)

if __name__ == "__main__":
    suspect = "I was at the store at 8 AM and talked to the cashier. I didn’t take anything."
    witness = "The suspect entered around 8 AM and took something without paying."
    print(generate_difference_report(suspect, witness))
