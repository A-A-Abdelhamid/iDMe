import numpy as np
import awkward as ak
import sys
sys.path.append("../../../analysisTools/")
import analysisSubroutines as routines

import vector

def cut5(events,info):
    # UL b-tag threshold recommendations from here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
    # using the medium WP, as in Andre's version of iDM
    name = "cut5"
    desc = "No b-tagged jets"
    plots = False
    bTag = events.PFJet.bTag
    # DeepFlavour working points for UL samples
    if info["year"] == 2018:
        # twiki : https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL18
        #wp = 0.0490 # loose
        wp = 0.2783 # medium
        #wp = 0.7100 # tight
    if info["year"] == 2017:
        # twiki : https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17
        #wp = 0.0532 # loose
        wp = 0.3040 # medium
        #wp = 0.7476 # tight
    if str(info["year"]) == "2016APV":
        # twiki : https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16preVFP
        #wp = 0.0508 # loose
        wp = 0.2598 # medium
        #wp = 0.6502 # tight
    if str(info["year"]) == "2016":
        # twiki : https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16postVFP
        #wp = 0.0480 # loose
        wp = 0.2489 # medium
        #wp = 0.6377 # tight
    pass_bTag = bTag > wp
    n_bTag_Jets = ak.count(events.PFJet[pass_bTag].pt,axis=1)
    cut = n_bTag_Jets == 0
    return events[cut], name, desc, plots

def cut6(events,info):
    name = "cut6"
    desc = "Leading jet pT > 80 GeV"
    plots = False
    cut = events.PFJet.pt[:,0] > 80
    return events[cut], name, desc, plots

def cut7(events,info):
    name = "cut7"
    desc = "Leading jet |eta| < 2.4"
    plots = False
    cut = np.abs(events.PFJet.eta[:,0]) < 2.4
    return events[cut], name, desc, plots

def cut8(events,info):
    name = "cut8"
    desc = "dPhi(MET,leading jet) > 1.5"
    plots = False
    cut = np.abs(events.PFJet.METdPhi[:,0]) > 1.5
    return events[cut], name, desc, plots

def cut9(events,info):
    name = "cut9"
    desc = "dPhi(MET,all jets) > 0.75"
    plots = False
    cut = ak.all(np.abs(events.PFJet.METdPhi) > 0.75,axis=1)
    return events[cut], name, desc, plots

def cut10(events,info):
    name = "cut10"
    desc = "OSSF"
    plots = False
    cut = events.sel_vtx.sign == -1
    return events[cut], name, desc, plots

def cut11(events,info):
    name = "cut11"
    desc = "cos(coll)"
    plots = False
    cut = events.sel_vtx.cos_collinear_fromPV_refit >= 0.4
    return events[cut], name, desc, plots

def cut12(events,info):
    name = "cut12"
    desc = "BDT"
    plots = True
    if len(events) != 0:
        score_BDT = events.BDTScore
        thres = 0.95
        cut = (score_BDT >= thres)

        print(f'BDT Pass: {np.count_nonzero(cut)}/{len(cut)}')
    else:
        cut = []
    
    return events[cut], name, desc, plots
"""
def cut13(events, info):
    name = "cut13"
    desc = "Reject events with ≥2 displaced standalone muons (pT>5 GeV, |eta|<2.4, displacedId)"
    plots = False

    required = ["recoDSAMuonPt", "recoDSAMuonEta", "recoDSAMuonDisplacedId"]
    if not all(hasattr(events, br) for br in required):
        # If DSA branches missing → nothing to veto
        return events, name, desc, plots

    pt  = events.recoDSAMuonPt
    eta = events.recoDSAMuonEta
    did = events.recoDSAMuonDisplacedId

    good_mu = (did == True) & (pt > 5.0) & (np.abs(eta) < 2.4)
    n_good_mu = ak.sum(good_mu, axis=1)

    # Keep events with 0 or 1 good DSA muon
    cut = n_good_mu < 2
    return events[cut], name, desc, plots
"""
"""
def cut14(events, info):
    name = "cut14"
    desc = "Reject events with ≥1 displaced standalone muon (pT>5 GeV, |eta|<2.4, displacedId)"
    plots = False

    required = ["recoDSAMuonPt", "recoDSAMuonEta", "recoDSAMuonDisplacedId"]
    if not all(hasattr(events, br) for br in required):
        # If DSA branches missing → nothing to veto
        return events, name, desc, plots

    pt  = events.recoDSAMuonPt
    eta = events.recoDSAMuonEta
    did = events.recoDSAMuonDisplacedId

    good_mu = (did == True) & (pt > 5.0) & (np.abs(eta) < 2.4)
    n_good_mu = ak.sum(good_mu, axis=1)

    # Keep only events with 0 good DSA muons
    cut = n_good_mu == 0
    return events[cut], name, desc, plots
"""

def cut15(events, info):
    name = "cut15"
    desc = (
        "Veto events with ≥2 displaced standalone muons (pT>5,|eta|<2.4,displacedId) "
        "that contain ≥1 OS DSA pair with ΔR<0.9 AND |Δφ(MET,pT^μμ)|<0.5"
    )
    plots = False

    # =====================================================
    # Step 0: require DSA muon branches
    # =====================================================
    required = ["recoDSAMuonPt", "recoDSAMuonEta", "recoDSAMuonDisplacedId"]
    if not all(hasattr(events, br) for br in required):
        return events, name, desc, plots

    pt  = events.recoDSAMuonPt
    eta = events.recoDSAMuonEta
    phi = events.recoDSAMuonPhi
    q   = events.recoDSAMuonCharge
    did = events.recoDSAMuonDisplacedId

    # =====================================================
    # Step 1: Identify "good" DSA muons
    # =====================================================
    good_mu = (did == True) & (pt > 5.0) & (np.abs(eta) < 2.4)
    n_good  = ak.sum(good_mu, axis=1)

    # Only events with >=2 good DSA muons are *candidates* for veto
    candidate = (n_good >= 2)

    # If none qualify, nothing to veto
    if not ak.any(candidate):
        return events, name, desc, plots

    # Restrict arrays to good muons only inside candidate events
    evt = events[candidate]

    mu_eta = evt.recoDSAMuonEta[good_mu[candidate]]
    mu_phi = evt.recoDSAMuonPhi[good_mu[candidate]]
    mu_pt  = evt.recoDSAMuonPt [good_mu[candidate]]
    mu_q   = evt.recoDSAMuonCharge[good_mu[candidate]]

    
    
    # =====================================================
    # Step 2: Build OS pairs using index-based combinations
    # =====================================================

    # Local index per muon in each candidate event
    idx = ak.local_index(mu_eta)  # jagged: [event, muon index]

    pairs = ak.combinations(idx, 2, axis=1, fields=["i1", "i2"])

    i1 = pairs.i1
    i2 = pairs.i2

    eta1 = mu_eta[i1]
    eta2 = mu_eta[i2]

    phi1 = mu_phi[i1]
    phi2 = mu_phi[i2]

    pt1  = mu_pt[i1]
    pt2  = mu_pt[i2]

    q1   = mu_q[i1]
    q2   = mu_q[i2]

    # Opposite-sign selection
    os_mask = (q1 != q2)

    eta1 = eta1[os_mask]
    eta2 = eta2[os_mask]
    phi1 = phi1[os_mask]
    phi2 = phi2[os_mask]
    pt1  = pt1[os_mask]
    pt2  = pt2[os_mask]
    q1   = q1[os_mask]
    q2   = q2[os_mask]

    # =====================================================
    # Step 3: Compute ΔR
    # =====================================================
    deta = eta1 - eta2
    dphi_raw = phi1 - phi2
    dphi = (dphi_raw + np.pi) % (2*np.pi) - np.pi
    dR = np.sqrt(deta**2 + dphi**2)

    # =====================================================
    # Step 4: Δφ(MET, pT^μμ)
    # =====================================================
    met_phi = evt.PFMET.phi

    px = pt1*np.cos(phi1) + pt2*np.cos(phi2)
    py = pt1*np.sin(phi1) + pt2*np.sin(phi2)
    phi_dimu = np.arctan2(py, px)

    met_phi_b, phi_dimu_b = ak.broadcast_arrays(met_phi, phi_dimu)

    dphi_met = (met_phi_b - phi_dimu_b + np.pi) % (2*np.pi) - np.pi

    # =====================================================
    # Step 5: Veto logic
    # =====================================================
    veto_pair = (dR < 0.9) & (np.abs(dphi_met) < 0.5)
    veto_event = ak.any(veto_pair, axis=1)

    # =====================================================
    # Step 6: Map veto results back to full event list
    # =====================================================
    idx = ak.where(candidate)[0]   # numeric indices of candidate events

    keep = np.ones(len(events), dtype=bool)
    keep[idx] = ~ak.to_numpy(veto_event)

    return events[keep], name, desc, plots