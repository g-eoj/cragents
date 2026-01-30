const readme = await Deno.readTextFile("../README.md");

const frontMatter = `---
title: cragents
layout: layout.vto
---

`;

await Deno.writeTextFile("index.md", frontMatter + readme);
console.log("Generated index.md from README.md");
