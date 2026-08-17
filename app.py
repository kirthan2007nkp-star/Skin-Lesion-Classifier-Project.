# ---------------------------------------------------------
# STEP-BY-STEP AI ANIMATION
# ---------------------------------------------------------
st.markdown("""
<style>
.workflow-container {
    margin: 25px 0;
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(128,128,128,0.25);
    overflow: hidden;
}

.workflow-title {
    text-align: center;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 25px;
}

.workflow {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
}

.workflow-step {
    flex: 1;
    text-align: center;
    padding: 18px 8px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.25);
    opacity: 0.25;
    transform: scale(0.92);
    animation: stepAnimation 12s infinite;
}

.workflow-step:nth-child(1) {
    animation-delay: 0s;
}

.workflow-step:nth-child(2) {
    animation-delay: 2s;
}

.workflow-step:nth-child(3) {
    animation-delay: 4s;
}

.workflow-step:nth-child(4) {
    animation-delay: 6s;
}

.workflow-step:nth-child(5) {
    animation-delay: 8s;
}

.workflow-step:nth-child(6) {
    animation-delay: 10s;
}

.workflow-icon {
    font-size: 2.2rem;
    margin-bottom: 8px;
}

.workflow-name {
    font-weight: 700;
    font-size: 1rem;
}

.workflow-desc {
    font-size: 0.8rem;
    opacity: 0.7;
    margin-top: 5px;
}

.arrow {
    font-size: 1.5rem;
    opacity: 0.5;
}

@keyframes stepAnimation {

    0% {
        opacity: 0.25;
        transform: scale(0.92);
    }

    8% {
        opacity: 1;
        transform: scale(1.05);
    }

    18% {
        opacity: 1;
        transform: scale(1);
    }

    25% {
        opacity: 0.25;
        transform: scale(0.92);
    }

    100% {
        opacity: 0.25;
        transform: scale(0.92);
    }
}

@media (max-width: 800px) {

    .workflow {
        flex-direction: column;
    }

    .workflow-step {
        width: 90%;
    }

    .arrow {
        transform: rotate(90deg);
    }
}
</style>

<div class="workflow-container">

<div class="workflow-title">
🧠 How SkinGuard AI Analyzes Your Image
</div>

<div class="workflow">

<div class="workflow-step">
<div class="workflow-icon">📤</div>
<div class="workflow-name">Upload</div>
<div class="workflow-desc">
Skin-lesion image
</div>
</div>

<div class="arrow">➜</div>

<div class="workflow-step">
<div class="workflow-icon">🖼️</div>
<div class="workflow-name">Preprocess</div>
<div class="workflow-desc">
Resize to 224 × 224
</div>
</div>

<div class="arrow">➜</div>

<div class="workflow-step">
<div class="workflow-icon">🧠</div>
<div class="workflow-name">CNN Analysis</div>
<div class="workflow-desc">
Extract features
</div>
</div>

<div class="arrow">➜</div>

<div class="workflow-step">
<div class="workflow-icon">🔬</div>
<div class="workflow-name">Classification</div>
<div class="workflow-desc">
Analyze 7 classes
</div>
</div>

<div class="arrow">➜</div>

<div class="workflow-step">
<div class="workflow-icon">📊</div>
<div class="workflow-name">Probability</div>
<div class="workflow-desc">
Calculate confidence
</div>
</div>

<div class="arrow">➜</div>

<div class="workflow-step">
<div class="workflow-icon">🎯</div>
<div class="workflow-name">Result</div>
<div class="workflow-desc">
Final prediction
</div>
</div>

</div>
</div>
""", unsafe_allow_html=True)

