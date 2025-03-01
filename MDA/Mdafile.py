import os
import requests
import MDAnalysis as mda
from rdkit import Chem
from rdkit.Chem import AllChem
from prolif import Molecule, InteractionFingerprint

# --- Step 1: Download PDB File ---
def download_pdb(pdb_id, save_path="structure.pdb"):
    """Download a PDB file from RCSB PDB."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "w") as file:
            file.write(response.text)
        print(f"Downloaded PDB: {save_path}")
    else:
        print("Error: Invalid PDB ID or network issue.")

# --- Step 2: Parse PDB (Handling Heavy-Atom-Only Structures) ---
def parse_pdb(pdb_file):
    """Load PDB file and handle heavy-atom-only structures."""
    u = mda.Universe(pdb_file)
    print("PDB Loaded Successfully!")
    return u

# --- Step 3: Infer Hydrogen Bonds from Heavy Atoms ---
def detect_hbonds(pdb_file):
    """Identify hydrogen bonds without explicit hydrogen using RDKit SMARTS."""
    mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)

    if mol is None:
        print("Error: Could not parse molecule.")
        return

    mol = Chem.AddHs(mol)  # Consider implicit hydrogens
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # Generate 3D coordinates

    # SMARTS patterns for Donors and Acceptors
    donor_smarts = Chem.MolFromSmarts("[N,O;H1,H2]")
    acceptor_smarts = Chem.MolFromSmarts("[N,O;!H0]")

    donors = mol.GetSubstructMatches(donor_smarts)
    acceptors = mol.GetSubstructMatches(acceptor_smarts)

    print(f"Hydrogen Bond Donors: {len(donors)}")
    print(f"Hydrogen Bond Acceptors: {len(acceptors)}")

# --- Step 4: Analyze with ProLIF ---
def analyze_interactions(pdb_file):
    """Use ProLIF to detect interaction fingerprints."""
    mol = Molecule(pdb_file)
    print(f"Loaded Molecule: {mol}")

    # Interaction Fingerprint Analysis
    ifp = InteractionFingerprint()
    interactions = ifp.generate(mol)
    print("Generated Interaction Fingerprint:", interactions)

# --- Main Execution ---
if __name__ == "__main__":
    pdb_id = "1CRN"  # Change this to your desired PDB ID
    pdb_file = "structure.pdb"

    download_pdb(pdb_id, pdb_file)
    parse_pdb(pdb_file)
    detect_hbonds(pdb_file)
    analyze_interactions(pdb_file)
