const BASE_URL = "";

export async function searchVideos(query, numResults = 10) {
  const res = await fetch(`${BASE_URL}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, num_results: numResults }),
  });
  if (!res.ok) throw new Error(`Search failed: ${res.statusText}`);
  return res.json();
}

export async function listVideos() {
  const res = await fetch(`${BASE_URL}/api/videos`);
  if (!res.ok) throw new Error(`Failed to list videos: ${res.statusText}`);
  return res.json();
}

export function getThumbnailUrl(segmentId) {
  return `${BASE_URL}/api/thumbnail/${segmentId}`;
}

export function getYoutubeUrlWithTimestamp(youtubeUrl, startTime) {
  const seconds = Math.floor(startTime);
  return `${youtubeUrl}&t=${seconds}s`;
}
