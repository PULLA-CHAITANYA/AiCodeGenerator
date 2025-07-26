document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('generator-form');
    const generateBtn = document.getElementById('generate-btn');
    const loader = document.getElementById('loader');
    const resultsContainer = document.getElementById('results-container');
    const errorMessage = document.getElementById('error-message');
    const explainBtn = document.getElementById('explain-btn');
    const explanationText = document.getElementById('explanation-text');

    const recursiveCodeElement = document.getElementById('recursive-code');
    const nonRecursiveCodeElement = document.getElementById('non-recursive-code');

    form.addEventListener('submit', async function (event) {
        event.preventDefault();

        const prompt = document.getElementById('prompt').value;
        const language = document.getElementById('language').value.toLowerCase();

        loader.style.display = 'block';
        generateBtn.disabled = true;
        generateBtn.textContent = 'Generating...';
        resultsContainer.style.display = 'none';
        errorMessage.style.display = 'none';
        explanationText.style.display = 'none';
        explanationText.textContent = '';

        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt, language: document.getElementById('language').value })
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || `HTTP error! Status: ${response.status}`);
            }

            recursiveCodeElement.textContent = data.recursive_solution || "// Recursive code not found";
            recursiveCodeElement.className = `language-${language}`;

            nonRecursiveCodeElement.textContent = data.iterative_solution || "// Iterative code not found";
            nonRecursiveCodeElement.className = `language-${language}`;

            Prism.highlightAll();
            resultsContainer.style.display = 'flex';
            document.querySelector('.explain-section').style.display = 'block';

        } catch (error) {
            errorMessage.textContent = `${error.message}`;
            errorMessage.style.display = 'block';
        } finally {
            loader.style.display = 'none';
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate';
        }
    });

    explainBtn.addEventListener('click', async () => {
        const recursiveCode = recursiveCodeElement.textContent;
        const iterativeCode = nonRecursiveCodeElement.textContent;

        explanationText.textContent = "Explaining...";
        explanationText.style.display = 'block';

        try {
            const response = await fetch('/explain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ recursiveCode, iterativeCode })
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || `HTTP error! Status: ${response.status}`);
            }

            explanationText.textContent = data.explanation;

        } catch (error) {
            explanationText.textContent = "Error: " + error.message;
        }
    });

    document.querySelectorAll('.copy-btn').forEach(button => {
        button.addEventListener('click', () => {
            const targetId = button.dataset.target;
            const codeElement = document.getElementById(targetId);
            navigator.clipboard.writeText(codeElement.textContent).then(() => {
                button.textContent = 'Copied!';
                setTimeout(() => { button.textContent = 'Copy'; }, 2000);
            });
        });
    });
});
