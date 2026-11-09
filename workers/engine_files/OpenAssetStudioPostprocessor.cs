// Open Asset Studio: Unity import helper.   EXPERIMENTAL - not tested in the Unity Editor by its authors (no Unity on the build machine).
//
// Put this file in a folder named "Editor" anywhere under Assets/. It only touches models that have a manifest.json written by Open Asset
// Studio ("tool": "Open Asset Studio") in the same folder; every other import is left alone.
//   * every child named UCX_*  becomes a convex MeshCollider (its renderer is removed, so the hull does not draw)
//   * a LOD Group created from the Name_LOD0..n children gets the transition heights of manifest.json (lods[].screen_size)
#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace OpenAssetStudio
{
    [Serializable] class LodInfo { public string name; public float screen_size; }
    [Serializable] class Manifest { public string tool; public LodInfo[] lods; }

    public class OpenAssetStudioPostprocessor : AssetPostprocessor
    {
        static Manifest ReadManifest(string modelPath)
        {
            var file = Path.Combine(Path.GetDirectoryName(modelPath) ?? "", "manifest.json");
            if (!File.Exists(file)) return null;
            try
            {
                var m = JsonUtility.FromJson<Manifest>(File.ReadAllText(file));
                return m != null && m.tool == "Open Asset Studio" ? m : null;
            }
            catch (Exception) { return null; }
        }

        void OnPostprocessModel(GameObject root)
        {
            var manifest = ReadManifest(assetPath);
            if (manifest == null) return;

            var hulls = new List<Transform>();
            foreach (var t in root.GetComponentsInChildren<Transform>(true))
                if (t.name.StartsWith("UCX_", StringComparison.Ordinal) || t.name.StartsWith("UBX_", StringComparison.Ordinal)) hulls.Add(t);
            foreach (var t in hulls)
            {
                var filter = t.GetComponent<MeshFilter>();
                if (filter == null || filter.sharedMesh == null) continue;
                var collider = t.gameObject.AddComponent<MeshCollider>();
                collider.sharedMesh = filter.sharedMesh;
                collider.convex = true;
                var renderer = t.GetComponent<MeshRenderer>();
                if (renderer != null) UnityEngine.Object.DestroyImmediate(renderer);
            }

            if (manifest.lods == null || manifest.lods.Length < 2) return;
            foreach (var group in root.GetComponentsInChildren<LODGroup>(true))
            {
                var lods = group.GetLODs();
                for (int i = 0; i < lods.Length && i < manifest.lods.Length; i++)
                    lods[i].screenRelativeTransitionHeight = manifest.lods[i].screen_size;
                group.SetLODs(lods);
                group.RecalculateBounds();
            }
        }
    }
}
#endif
