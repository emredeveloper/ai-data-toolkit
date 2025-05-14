chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.type === "GET_PAGE_TEXT") {
      const bodyText = document.body.innerText.trim();
      sendResponse({ text: bodyText || null });
    }
});